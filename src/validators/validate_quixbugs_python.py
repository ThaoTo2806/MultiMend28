import contextlib
import json
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


# ============================================================
# CONFIG
# ============================================================

project_dir = quixbugs_dir
gen_dir = quixbugs_genpy_dir

bugs_metadata_file = "QuixBugs_Python.jsonl"

model = "multimend"

output_dir = gen_dir / f"outputs-{model}"

temp_dir = output_dir / "temp"
save_state_dir = output_dir / "save-state"

output_size = 100

N_JOBS = 6
TEST_TIMEOUT = 60
VALIDATION_SLEEP = 1.0


# ============================================================
# STATUS
# ============================================================

class Status(Enum):
    PLAUSIBLE = auto()
    PARSABLE = auto()
    TIMEOUT = auto()
    UNPARSABLE = auto()


# ============================================================
# TQDM / JOBLIB
# ============================================================

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager that redirects joblib progress
    into a tqdm progress bar.
    """

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = (
                self.n_completed_tasks
                - tqdm_object.n
            )
            tqdm_object.update(
                n=n_completed
            )

    original_print_progress = (
        joblib.parallel.Parallel.print_progress
    )

    joblib.parallel.Parallel.print_progress = (
        tqdm_print_progress
    )

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = (
            original_print_progress
        )
        tqdm_object.close()


# ============================================================
# DATAFRAME HELPERS
# ============================================================

def get_hunk_candidates(
    df: pd.DataFrame,
    hunk: int,
) -> pd.DataFrame:
    """
    Return candidate patches for a specific hunk.
    """

    return df.loc[
        df["hunk"] == hunk
    ]


def get_candidates(
    df: pd.DataFrame,
    bugid: str,
) -> pd.DataFrame:
    """
    Return all candidates for a specific bug.

    astype(str) is used only to avoid mismatches between
    JSON metadata and pandas string/numeric representations.
    """

    return df.loc[
        df["bugid"].astype(str)
        == str(bugid)
    ]


# ============================================================
# PATCH INSERTION
# ============================================================

def insert_patch(
    patch,
    source_file_path,
    target_file_path,
    bug_line,
    bug_len,
    indent,
):
    """
    Insert one generated patch into a clean copy
    of the buggy program.

    This follows the original validator behavior.
    """

    with open(
        source_file_path,
        "r",
    ) as file:
        lines = file.readlines()

    patch = str(patch)

    if bug_len == 0:

        lines.insert(
            bug_line,
            indent + patch + "\n",
        )

    else:

        lines[
            bug_line - 1:
            (bug_line - 1) + bug_len
        ] = (
            indent + patch + "\n"
        )

    with open(
        target_file_path,
        "w",
    ) as file:
        file.writelines(lines)


# ============================================================
# COPY DATASET
# ============================================================

def copy_dataset_files(
    dataset_dir,
    temp_dataset_dir,
):
    """
    Copy the complete QuixBugs dataset to a worker-local
    temporary directory.
    """

    shutil.copytree(
        dataset_dir,
        temp_dataset_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            ".*"
        ),
    )


# ============================================================
# TEST EXECUTION
# ============================================================

def run_tests(
    bugid: str,
    project_copy_dir: Path,
) -> Status:
    """
    Run the QuixBugs test corresponding to bugid.

    Return-code protocol:

        0 -> PLAUSIBLE
        2 -> UNPARSABLE
        other non-zero -> PARSABLE
        timeout -> TIMEOUT
    """

    tests_dir = (
        project_copy_dir
        / "python_testcases"
    )

    test_file = (
        tests_dir
        / f"test_{bugid}.py"
    )

    args = [
        "pytest",
        "-x",
        str(test_file),
    ]

    try:

        result = subprocess.run(
            args,
            capture_output=True,
            timeout=TEST_TIMEOUT,
        )

    except subprocess.TimeoutExpired:

        return Status.TIMEOUT

    if result.returncode == 0:

        return Status.PLAUSIBLE

    elif result.returncode == 2:

        return Status.UNPARSABLE

    else:

        return Status.PARSABLE


# ============================================================
# VALIDATE ONE BUG
# ============================================================

def apply_patch(
    cp_df: pd.DataFrame,
    bugid: str,
    hunks: list,
) -> Optional[pd.DataFrame]:
    """
    Validate generated patches for one bug.

    This intentionally follows the original MultiMend
    QuixBugs validator:

    - only single-hunk bugs are validated here;
    - candidates are taken from hunk 0;
    - candidates are evaluated in DataFrame order;
    - validation stops at the first plausible candidate;
    - timeout is also marked as parsable;
    - validation state is saved per bug.
    """

    save_file_path = (
        save_state_dir
        / f"{bugid}.jsonl"
    )

    # --------------------------------------------------------
    # Already processed
    # --------------------------------------------------------

    if save_file_path.exists():
        return None

    # --------------------------------------------------------
    # Worker ID
    # --------------------------------------------------------

    pid = threading.get_ident()

    # --------------------------------------------------------
    # Original MultiMend validator handles single-hunk bugs.
    # --------------------------------------------------------

    if len(hunks) != 1:
        return None

    hunk = hunks[0]

    # --------------------------------------------------------
    # Worker-local QuixBugs copy
    # --------------------------------------------------------

    project_copy_dir = (
        temp_dir
        / str(pid)
        / "QuixBugs"
    )

    copy_dataset_files(
        project_dir,
        project_copy_dir,
    )

    # --------------------------------------------------------
    # Target program
    # --------------------------------------------------------

    target_file_path = (
        project_copy_dir
        / "python_programs"
        / f"{bugid}.py"
    )

    # --------------------------------------------------------
    # Original buggy program
    # --------------------------------------------------------

    source_file_path = (
        temp_dir
        / str(pid)
        / "sources"
        / bugid
        / f"{bugid}.py"
    )

    source_file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copyfile(
        target_file_path,
        source_file_path,
    )

    # --------------------------------------------------------
    # Hunk information
    # --------------------------------------------------------

    bug_line, bug_len = (
        hunk[
            "removed_line_numbers_range"
        ]
    )

    # --------------------------------------------------------
    # Original implementation validates hunk 0.
    # --------------------------------------------------------

    bug_hunk_subset_df = (
        get_hunk_candidates(
            cp_df,
            0,
        )
    )

    # --------------------------------------------------------
    # Preserve indentation
    # --------------------------------------------------------

    added_lines = hunk.get(
        "added_lines",
        "",
    )

    indent_size = (
        len(added_lines)
        - len(
            added_lines.lstrip(
                " \t"
            )
        )
    )

    indent = added_lines[
        :indent_size
    ]

    # --------------------------------------------------------
    # Validate candidates
    # --------------------------------------------------------

    for index, patch in (
        bug_hunk_subset_df[
            "decoded_sequences"
        ].items()
    ):

        # ----------------------------------------------------
        # Apply candidate to CLEAN source.
        #
        # insert_patch() always reads source_file_path,
        # therefore previous candidates do not accumulate.
        # ----------------------------------------------------

        insert_patch(
            patch,
            source_file_path,
            target_file_path,
            bug_line,
            bug_len,
            indent,
        )

        # ----------------------------------------------------
        # Run tests
        # ----------------------------------------------------

        start_timer = (
            timeit.default_timer()
        )

        passed = run_tests(
            bugid,
            project_copy_dir,
        )

        end_timer = (
            timeit.default_timer()
        )

        cp_df.at[
            index,
            "validation_time",
        ] = (
            end_timer
            - start_timer
        )

        # ----------------------------------------------------
        # PLAUSIBLE
        # ----------------------------------------------------

        if passed is Status.PLAUSIBLE:

            cp_df.at[
                index,
                "plausible",
            ] = True

            cp_df.at[
                index,
                "parsable",
            ] = True

            # Original behavior:
            # stop at first plausible candidate.

            break

        # ----------------------------------------------------
        # PARSABLE
        # ----------------------------------------------------

        elif passed is Status.PARSABLE:

            cp_df.at[
                index,
                "parsable",
            ] = True

        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        elif passed is Status.TIMEOUT:

            cp_df.at[
                index,
                "timeout",
            ] = True

            cp_df.at[
                index,
                "parsable",
            ] = True

        # ----------------------------------------------------
        # UNPARSABLE
        #
        # Nothing else is changed.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Original implementation sleeps between candidates.
        # ----------------------------------------------------

        time.sleep(
            VALIDATION_SLEEP
        )

    # --------------------------------------------------------
    # Save validation state
    # --------------------------------------------------------

    save_state_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cp_df.to_json(
        save_state_dir
        / f"{bugid}.jsonl",
        orient="records",
        lines=True,
    )

    return cp_df


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():
    """
    Load QuixBugs-Python metadata.
    """

    with open(
        gen_dir
        / bugs_metadata_file
    ) as meta_file:

        bugs_metadata = ChainMap(
            *[
                json.loads(line)
                for line in meta_file
            ][::-1]
        )

    return bugs_metadata


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MultiMend QuixBugs-Python Validator"
    )

    print(
        "=" * 60
    )

    print(
        f"Project directory : {project_dir}"
    )

    print(
        f"Output directory  : {output_dir}"
    )

    print(
        f"Candidate size    : {output_size}"
    )

    print(
        f"Workers           : {N_JOBS}"
    )

    print(
        f"Test timeout      : {TEST_TIMEOUT}s"
    )

    # ========================================================
    # LOAD METADATA
    # ========================================================

    bugs_metadata = load_metadata()

    print(
        f"Metadata bugs     : "
        f"{len(bugs_metadata)}"
    )

    # ========================================================
    # LOAD CANDIDATES
    # ========================================================

    candidate_file = (
        output_dir
        / f"final_candidates_{output_size}.jsonl"
    )

    if not candidate_file.exists():

        raise FileNotFoundError(
            f"Candidate file not found:\n"
            f"{candidate_file}"
        )

    candidate_patches_df = pd.read_json(
        candidate_file,
        orient="records",
        lines=True,
    )

    print(
        f"Candidate rows    : "
        f"{len(candidate_patches_df)}"
    )

    # ========================================================
    # VALIDATION COLUMNS
    # ========================================================

    candidate_patches_df[
        "plausible"
    ] = False

    candidate_patches_df[
        "parsable"
    ] = False

    candidate_patches_df[
        "timeout"
    ] = False

    candidate_patches_df[
        "validation_time"
    ] = np.nan

    # ========================================================
    # CLEAN TEMP DIRECTORY
    # ========================================================

    shutil.rmtree(
        temp_dir,
        ignore_errors=True,
    )

    save_state_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    with tqdm_joblib(
        tqdm(
            total=len(bugs_metadata),
            desc="Validating bugs",
        )
    ):

        Parallel(
            n_jobs=N_JOBS,
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
            for bugid, hunks
            in bugs_metadata.items()
        )

    # ========================================================
    # LOAD VALIDATION STATES
    # ========================================================

    state_files = sorted(
        save_state_dir.iterdir()
    )

    cp_dfs = [
        pd.read_json(
            state_file,
            orient="records",
            lines=True,
        )
        for state_file in state_files
        if state_file.is_file()
    ]

    if not cp_dfs:

        raise RuntimeError(
            "No validation state files were produced."
        )

    concatenated_cp_df = pd.concat(
        cp_dfs,
        ignore_index=True,
    )

    # ========================================================
    # CHECK CANDIDATE COUNT
    # ========================================================

    assert (
        len(candidate_patches_df)
        == len(concatenated_cp_df)
    ), (
        "Candidate count mismatch: "
        f"{len(candidate_patches_df)} "
        "vs "
        f"{len(concatenated_cp_df)}"
    )

    # ========================================================
    # SAVE CANDIDATE-LEVEL RESULTS
    # ========================================================

    result_file = (
        output_dir
        / f"plausible_candidates_{output_size}.jsonl"
    )

    concatenated_cp_df.to_json(
        result_file,
        orient="records",
        lines=True,
    )

    print(
        "\nSaved candidate-level results:"
    )

    print(
        result_file
    )

    # ========================================================
    # PAPER / ORIGINAL PLAUSIBLE METRIC
    # ========================================================
    #
    # A bug is plausible iff every hunk has at least one
    # plausible candidate.
    #
    # Equivalent to:
    #
    # groupby(["bugid", "hunk"]).plausible.any()
    #       .groupby("bugid").all()
    #
    # This is kept exactly as in the supplied original code.
    # ========================================================

    bugs_with_plausible_patch = (
        concatenated_cp_df
        .groupby(
            [
                "bugid",
                "hunk",
            ]
        )[
            "plausible"
        ]
        .any()
        .groupby(
            "bugid"
        )
        .all()
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MultiMend PLAUSIBLE RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        bugs_with_plausible_patch
    )

    print(
        "\nValue counts:"
    )

    print(
        bugs_with_plausible_patch
        .value_counts()
    )

    print(
        "\nTotal bugs:"
        f" {len(bugs_with_plausible_patch)}"
    )

    print(
        "Plausible bugs:"
        f" "
        f"{int(bugs_with_plausible_patch.sum())}"
    )

    # ========================================================
    # SAVE BUG-LEVEL RESULTS
    # ========================================================

    bug_metrics_file = (
        output_dir
        / f"plausible_bugs_{output_size}.jsonl"
    )

    bug_metrics = (
        bugs_with_plausible_patch
        .rename("plausible")
        .reset_index()
    )

    bug_metrics.to_json(
        bug_metrics_file,
        orient="records",
        lines=True,
    )

    print(
        "\nSaved bug-level results:"
    )

    print(
        bug_metrics_file
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()