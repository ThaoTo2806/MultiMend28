
import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
import timeit
from collections import ChainMap
from copy import deepcopy
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from ..configs import quixbugs_dir, quixbugs_genpy_dir


project_dir = quixbugs_dir
gen_dir = quixbugs_genpy_dir
bugs_metadata_file = "QuixBugs_Python.jsonl"
model = "multimend"

context_strategy = os.environ.get("MULTIMEND_CONTEXT_STRATEGY", "fixed_rag")
if context_strategy not in {"no_rag", "fixed_rag"}:
    raise ValueError(f"Unsupported context strategy: {context_strategy}")

output_dir = gen_dir / f"outputs-{model}-{context_strategy}"
temp_dir = output_dir / "temp"
save_state_dir = output_dir / "save-state"
output_size = 100

# Configurable through environment variables.
# Validation configuration.
# Default timeout remains 60 seconds, but can be overridden by environment.
DEFAULT_TEST_TIMEOUT = float(
    os.environ.get("MULTIMEND_VALIDATION_TIMEOUT", "60")
)

CANDIDATE_SLEEP = float(
    os.environ.get("MULTIMEND_VALIDATION_SLEEP", "1.0")
)

N_WORKERS = int(
    os.environ.get("MULTIMEND_VALIDATION_JOBS", "6")
)

RESET_VALIDATION = os.environ.get(
    "MULTIMEND_RESET_VALIDATION", "0"
).lower() in {"1", "true", "yes"}

rem_file_path = gen_dir / "rem.txt"
add_file_path = gen_dir / "add.txt"

with (
    open(rem_file_path) as rem_file,
    open(add_file_path) as add_file,
):
    sources = [src.strip() for src in rem_file]
    targets = [tgt.strip() for tgt in add_file]


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()


def get_hunk_candidates(df: pd.DataFrame, hunk: int) -> pd.DataFrame:
    """Returns candidate patches for a specific hunk."""
    return df.loc[df["hunk"] == hunk]


def get_candidates(df: pd.DataFrame, bugid: str) -> pd.DataFrame:
    """Returns candidate patches for a specific bugid."""
    return df.loc[df["bugid"] == bugid]


def insert_patch(patch, source_file_path, target_file_path, bug_line, bug_len, indent):
    with open(source_file_path, "r") as file:
        lines = file.readlines()

    if bug_len == 0:
        lines.insert(bug_line, indent + patch + "\n")
    else:
        lines[bug_line - 1 : (bug_line - 1) + bug_len] = [indent + patch + "\n"]

    with open(target_file_path, "w") as file:
        file.writelines(lines)


class Status(Enum):
    PLAUSIBLE = auto()
    PARSABLE = auto()
    TIMEOUT = auto()
    UNPARSABLE = auto()


def run_tests(bugid: str, project_copy_dir: Path) -> Status:
    """Run pytest for one candidate patch."""

    timeout = DEFAULT_TEST_TIMEOUT

    tests_dir = project_copy_dir / "python_testcases"
    test_file = f"test_{bugid}.py"

    args = [
        "pytest",
        "-x",
        str(tests_dir / test_file),
    ]

    process = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return Status.TIMEOUT

    if process.returncode == 0:
        return Status.PLAUSIBLE
    elif process.returncode == 2:
        return Status.UNPARSABLE
    else:
        return Status.PARSABLE


def apply_patch(
    cp_df: pd.DataFrame,
    bugid: str,
    hunks: list,
) -> Optional[pd.DataFrame]:

    save_file_path = save_state_dir / f"{bugid}.jsonl"

    # ---------------------------------------------------------
    # Load old save-state and preserve the separately computed metrics.
    # ---------------------------------------------------------
    if save_file_path.exists():
        saved_df = pd.read_json(
            save_file_path,
            orient="records",
            lines=True,
        )

        if "exact_match" in saved_df.columns:
            saved_df["exact_match"] = (
                saved_df["exact_match"].fillna(False).astype(bool)
            )
        saved_df.drop(columns=["correct"], errors="ignore", inplace=True)
        if "paper_correct" not in saved_df.columns:
            saved_df["paper_correct"] = False
        save_validation_state(saved_df, bugid)

        return saved_df

    pid = threading.get_ident()

    if len(hunks) == 1:
        hunk = hunks[0]

        project_copy_dir = temp_dir / str(pid) / "QuixBugs"
        copy_dataset_files(project_dir, project_copy_dir)

        target_file_path = project_copy_dir / "python_programs" / f"{bugid}.py"

        bug_line, bug_len = hunk["removed_line_numbers_range"]
        bug_hunk_subset_df = get_hunk_candidates(cp_df, 0)

        source_file_path = (
            temp_dir / str(pid) / "sources" / bugid / f"{bugid}.py"
        )
        source_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target_file_path, source_file_path)

        indent_size = len(hunk["added_lines"]) - len(
            hunk["added_lines"].lstrip(" \t")
        )
        indent = hunk["added_lines"][:indent_size]

        if "exact_match" in cp_df.columns:
            cp_df["exact_match"] = (
                cp_df["exact_match"].fillna(False).astype(bool)
            )
        cp_df.drop(columns=["correct"], errors="ignore", inplace=True)
        if "paper_correct" not in cp_df.columns:
            cp_df["paper_correct"] = False

        print(
            f"[VALIDATOR] bug={bugid} candidates={len(bug_hunk_subset_df)}",
            flush=True,
        )

        for index, patch in bug_hunk_subset_df["decoded_sequences"].items():

            print(
                f"[VALIDATOR] START index={index}",
                flush=True,
            )

            # Restore original source before applying candidate.
            shutil.copyfile(source_file_path, target_file_path)

            insert_patch(
                patch,
                source_file_path,
                target_file_path,
                bug_line,
                bug_len,
                indent,
            )

            start_timer = timeit.default_timer()

            passed = run_tests(
                bugid,
                project_copy_dir,
            )

            end_timer = timeit.default_timer()

            cp_df.at[index, "validation_time"] = (
                end_timer - start_timer
            )

            if passed is Status.PLAUSIBLE:
                cp_df.at[index, "plausible"] = True
                cp_df.at[index, "parsable"] = True
                break

            elif passed is Status.PARSABLE:
                cp_df.at[index, "parsable"] = True

            elif passed is Status.TIMEOUT:
                cp_df.at[index, "timeout"] = True
                cp_df.at[index, "parsable"] = True

            # Keep partial progress if a long-running bug is interrupted.
            save_validation_state(cp_df, bugid)

            # Configurable sleep between candidates.
            time.sleep(CANDIDATE_SLEEP)

        # Save intermediate state.
        save_validation_state(cp_df, bugid)

        return cp_df

    return cp_df


def copy_dataset_files(dataset_dir, temp_dataset_dir):
    shutil.copytree(
        dataset_dir,
        temp_dataset_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".*"),
    )


def save_validation_state(df: pd.DataFrame, bugid: str) -> None:
    """Persist a bug's state so interrupted validation can resume safely."""
    save_state_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_state_dir / f"{bugid}.jsonl"
    temporary_path = save_path.with_suffix(".jsonl.tmp")
    df.to_json(
        temporary_path,
        orient="records",
        lines=True,
    )
    temporary_path.replace(save_path)


def reset_stale_validation_state(candidate_file: Path) -> None:
    """Discard state produced from an older candidate file."""
    if not save_state_dir.exists():
        return

    candidate_mtime = candidate_file.stat().st_mtime_ns
    state_files = list(save_state_dir.glob("*.jsonl"))
    latest_state_mtime = (
        max(path.stat().st_mtime_ns for path in state_files)
        if state_files
        else 0
    )
    if state_files and latest_state_mtime < candidate_mtime:
        shutil.rmtree(save_state_dir)


def main():

    n_jobs = N_WORKERS

    print(
        "[VALIDATOR CONFIG] "
        f"test_timeout={DEFAULT_TEST_TIMEOUT}s, "
        f"candidate_sleep={CANDIDATE_SLEEP}s, "
        f"workers={n_jobs}",
        flush=True,
    )

    with open(gen_dir / bugs_metadata_file) as meta_file:
        bugs_metadata = ChainMap(
            *[json.loads(line) for line in meta_file][::-1]
        )

    # Validate ALL programs.
    # No programs are skipped.

    metadata_programs = len(bugs_metadata)

    print(f"Metadata programs: {metadata_programs}")
    print(f"Programs to validate: {len(bugs_metadata)}")

    candidate_file = output_dir / f"final_candidates_{output_size}.jsonl"
    candidate_patches_df = pd.read_json(
        candidate_file,
        orient="records",
        lines=True,
    )

    if RESET_VALIDATION:
        shutil.rmtree(save_state_dir, ignore_errors=True)
    else:
        reset_stale_validation_state(candidate_file)

    # Always initialize these columns.
    candidate_patches_df["plausible"] = False
    if "paper_correct" not in candidate_patches_df.columns:
        candidate_patches_df["paper_correct"] = False
    candidate_patches_df["paper_correct"] = (
        candidate_patches_df["paper_correct"].fillna(False).astype(bool)
    )
    candidate_patches_df["parsable"] = False
    candidate_patches_df["timeout"] = False
    candidate_patches_df["validation_time"] = np.nan

    shutil.rmtree(
        temp_dir,
        ignore_errors=True,
    )

    save_state_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tqdm_joblib(tqdm(total=len(bugs_metadata))):
        Parallel(
            n_jobs=n_jobs,
            backend="threading",
        )(
            delayed(apply_patch)(
                deepcopy(
                    get_candidates(
                        candidate_patches_df,
                        bugid,
                    )
                ),
                bugid,
                hunks,
            )
            for bugid, hunks in bugs_metadata.items()
        )

    cp_dfs = []

    for cp in sorted(save_state_dir.iterdir()):

        if cp.suffix != ".jsonl":
            continue

        df = pd.read_json(
            cp,
            orient="records",
            lines=True,
        )

        if "exact_match" not in df.columns:
            df["exact_match"] = False
        df["exact_match"] = df["exact_match"].fillna(False).astype(bool)
        if "paper_correct" not in df.columns:
            df["paper_correct"] = False
        df["paper_correct"] = df["paper_correct"].fillna(False).astype(bool)

        cp_dfs.append(df)

    concatenated_cp_df = pd.concat(
        cp_dfs,
        ignore_index=True,
    )

    assert len(candidate_patches_df) == len(concatenated_cp_df)

    concatenated_cp_df.to_json(
        output_dir / f"plausible_candidates_{output_size}.jsonl",
        orient="records",
        lines=True,
    )

    # ---------------------------------------------------------
    # Plausible: patch passes pytest.
    # ---------------------------------------------------------
    bugs_with_plausible_patch = (
        concatenated_cp_df
        .groupby(["bugid", "hunk"])["plausible"]
        .any()
        .groupby("bugid")
        .all()
    )

    print("\n===== PLAUSIBLE =====")
    print(bugs_with_plausible_patch)
    print(bugs_with_plausible_patch.value_counts())

    # ---------------------------------------------------------
    # Exact match: candidate matches the normalized developer patch.
    # ---------------------------------------------------------
    bugs_with_exact_patch = (
        concatenated_cp_df
        .groupby(["bugid", "hunk"])["exact_match"]
        .any()
        .groupby("bugid")
        .all()
    )

    print("\n===== EXACT MATCH =====")
    print(bugs_with_exact_patch)
    print(bugs_with_exact_patch.value_counts())

    # Paper correctness is a separate manual annotation and may include
    # semantically equivalent patches.
    bugs_with_paper_correct_patch = (
        concatenated_cp_df
        .groupby(["bugid", "hunk"])["paper_correct"]
        .any()
        .groupby("bugid")
        .all()
    )

    print("\n===== PAPER CORRECT (MANUAL) =====")
    print(bugs_with_paper_correct_patch)
    print(bugs_with_paper_correct_patch.value_counts())


if __name__ == "__main__":
    main()
