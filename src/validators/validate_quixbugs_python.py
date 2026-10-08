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


# ============================================================
# CONFIG
# ============================================================

project_dir = quixbugs_dir
gen_dir = quixbugs_genpy_dir

bugs_metadata_file = "QuixBugs_Python.jsonl"

model = "multimend"

# Cell 7 sets:
#   MULTIMEND_CONTEXT_STRATEGY=no_rag
#   MULTIMEND_CONTEXT_STRATEGY=fixed_rag
#
# Therefore validation output is separated by strategy.
context_strategy = os.environ.get(
    "MULTIMEND_CONTEXT_STRATEGY",
    "fixed_rag",
)

if context_strategy not in {"no_rag", "fixed_rag"}:
    raise ValueError(
        f"Unsupported context strategy: {context_strategy}"
    )

output_dir = (
    gen_dir
    / f"outputs-{model}-{context_strategy}"
)

temp_dir = output_dir / "temp"
save_state_dir = output_dir / "save-state"

output_size = 100

N_JOBS = int(
    os.environ.get(
        "MULTIMEND_VALIDATION_JOBS",
        "6",
    )
)

TEST_TIMEOUT = int(
    os.environ.get(
        "MULTIMEND_VALIDATION_TIMEOUT",
        "60",
    )
)

VALIDATION_SLEEP = float(
    os.environ.get(
        "MULTIMEND_VALIDATION_SLEEP",
        "1.0",
    )
)

RESET_VALIDATION = (
    os.environ.get(
        "MULTIMEND_RESET_VALIDATION",
        "0",
    )
    == "1"
)


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

def get_candidates(
    df: pd.DataFrame,
    bugid: str,
) -> pd.DataFrame:

    return df.loc[
        df["bugid"].astype(str)
        == str(bugid)
    ]


def get_hunk_candidates(
    df: pd.DataFrame,
    hunk: int,
) -> pd.DataFrame:

    return df.loc[
        df["hunk"] == hunk
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
# DATASET COPY
# ============================================================

def copy_dataset_files(
    dataset_dir,
    temp_dataset_dir,
):

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

    if result.returncode == 2:

        return Status.UNPARSABLE

    return Status.PARSABLE


# ============================================================
# VALIDATE ONE BUG
# ============================================================

def apply_patch(
    cp_df: pd.DataFrame,
    bugid: str,
    hunks: list,
) -> Optional[pd.DataFrame]:

    save_file_path = (
        save_state_dir
        / f"{bugid}.jsonl"
    )

    if save_file_path.exists():

        return None

    # --------------------------------------------------------
    # The original MultiMend QuixBugs validator validates
    # single-hunk bugs.
    # --------------------------------------------------------

    if len(hunks) != 1:

        return None

    hunk = hunks[0]

    pid = threading.get_ident()

    # --------------------------------------------------------
    # Worker-local dataset
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
    # Program paths
    # --------------------------------------------------------

    target_file_path = (
        project_copy_dir
        / "python_programs"
        / f"{bugid}.py"
    )

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
    # Original validator uses hunk 0.
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

        insert_patch(
            patch,
            source_file_path,
            target_file_path,
            bug_line,
            bug_len,
            indent,
        )

        start_timer = (
            timeit.default_timer()
        )

        status = run_tests(
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

        if status is Status.PLAUSIBLE:

            cp_df.at[
                index,
                "plausible",
            ] = True

            cp_df.at[
                index,
                "parsable",
            ] = True

            # Original MultiMend behavior:
            # stop after first plausible patch.
            break

        # ----------------------------------------------------
        # PARSABLE
        # ----------------------------------------------------

        elif status is Status.PARSABLE:

            cp_df.at[
                index,
                "parsable",
            ] = True

        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        elif status is Status.TIMEOUT:

            cp_df.at[
                index,
                "timeout",
            ] = True

            cp_df.at[
                index,
                "parsable",
            ] = True

        time.sleep(
            VALIDATION_SLEEP
        )

    # --------------------------------------------------------
    # Save per-bug state
    # --------------------------------------------------------

    save_state_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cp_df.to_json(
        save_file_path,
        orient="records",
        lines=True,
    )

    return cp_df


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    with open(
        gen_dir
        / bugs_metadata_file
    ) as meta_file:

        return ChainMap(
            *[
                json.loads(line)
                for line in meta_file
            ][::-1]
        )


# ============================================================
# GOLD / MANUAL CORRECTNESS
# ============================================================

def load_gold_annotations():

    """
    Load manually reviewed correctness labels.

    Expected JSONL format:

    {
        "bugid": "h4iku",
        "correct": true
    }

    OR:

    {
        "bugid": "h4iku",
        "paper_correct": true
    }

    The validator accepts either field name.
    """

    annotation_file = os.environ.get(
        "MULTIMEND_PAPER_CORRECT_FILE"
    )

    if not annotation_file:

        print(
            "\nNo MULTIMEND_PAPER_CORRECT_FILE supplied."
        )

        print(
            "Gold/manual correctness will be unavailable."
        )

        return {}

    annotation_path = Path(
        annotation_file
    )

    if not annotation_path.is_file():

        raise FileNotFoundError(
            "Gold/manual correctness file not found:\n"
            f"{annotation_path}"
        )

    annotations = pd.read_json(
        annotation_path,
        orient="records",
        lines=True,
    )

    if "bugid" not in annotations.columns:

        raise ValueError(
            "Gold annotation file must contain "
            "'bugid'."
        )

    if "paper_correct" in annotations.columns:

        correct_column = "paper_correct"

    elif "correct" in annotations.columns:

        correct_column = "correct"

    else:

        raise ValueError(
            "Gold annotation file must contain "
            "'correct' or 'paper_correct'."
        )

    annotations = annotations[
        [
            "bugid",
            correct_column,
        ]
    ].copy()

    annotations["bugid"] = (
        annotations["bugid"]
        .astype(str)
    )

    annotations[correct_column] = (
        annotations[correct_column]
        .astype(bool)
    )

    duplicated = (
        annotations
        .groupby("bugid")[
            correct_column
        ]
        .nunique()
    )

    conflicting = duplicated[
        duplicated > 1
    ]

    if len(conflicting):

        raise ValueError(
            "Conflicting gold correctness labels "
            f"for bugids: "
            f"{list(conflicting.index)}"
        )

    annotations = (
        annotations
        .drop_duplicates(
            "bugid"
        )
    )

    return dict(
        zip(
            annotations["bugid"],
            annotations[correct_column],
        )
    )


# ============================================================
# EVALUATION LABELS
# ============================================================

def add_evaluation_labels(
    df: pd.DataFrame,
    gold_annotations: dict,
):

    # --------------------------------------------------------
    # Identical = generated patch exactly matches gold patch.
    # --------------------------------------------------------

    if (
        "normalized_patch" in df.columns
        and "normalized_target" in df.columns
    ):

        df["identical"] = (
            df["normalized_patch"]
            == df["normalized_target"]
        )

    elif (
        "decoded_sequences" in df.columns
        and "target" in df.columns
    ):

        df["identical"] = (
            df["decoded_sequences"]
            .astype(str)
            .str.strip()
            ==
            df["target"]
            .astype(str)
            .str.strip()
        )

    else:

        df["identical"] = False

    # --------------------------------------------------------
    # Correct = manual/gold paper annotation.
    #
    # IMPORTANT:
    #
    # correct != plausible
    #
    # A patch can pass tests but still be incorrect.
    # --------------------------------------------------------

    if gold_annotations:

        df["correct"] = (
            df["bugid"]
            .astype(str)
            .map(gold_annotations)
            .astype("boolean")
        )

    else:

        df["correct"] = pd.Series(
            pd.array(
                [pd.NA] * len(df),
                dtype="boolean",
            ),
            index=df.index,
        )

    # Keep paper_correct as an alias for compatibility.
    df["paper_correct"] = df["correct"]

    return df


# ============================================================
# BUG-LEVEL PAPER METRICS
# ============================================================

def calculate_bug_metrics(
    df: pd.DataFrame,
):

    # --------------------------------------------------------
    # PLAUSIBLE
    #
    # A bug is plausible when every hunk has at least one
    # plausible generated patch.
    # --------------------------------------------------------

    plausible_by_hunk = (
        df.groupby(
            [
                "bugid",
                "hunk",
            ]
        )[
            "plausible"
        ]
        .any()
    )

    plausible_by_bug = (
        plausible_by_hunk
        .groupby("bugid")
        .all()
    )

    # --------------------------------------------------------
    # CORRECT
    #
    # Correctness comes from gold/manual annotations.
    #
    # We DO NOT define:
    #
    #     correct = plausible
    #
    # and we DO NOT define:
    #
    #     correct = identical
    #
    # --------------------------------------------------------

    if (
        "correct" in df.columns
        and df["correct"].notna().any()
    ):

        correct_by_bug = (
            df.groupby("bugid")[
                "correct"
            ]
            .first()
        )

    else:

        correct_by_bug = pd.Series(
            pd.NA,
            index=plausible_by_bug.index,
            dtype="boolean",
        )

    # --------------------------------------------------------
    # Identical gold-patch metric
    # --------------------------------------------------------

    identical_by_bug = (
        df.groupby("bugid")[
            "identical"
        ]
        .any()
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    all_bugids = sorted(
        set(plausible_by_bug.index)
        | set(correct_by_bug.index)
        | set(identical_by_bug.index)
    )

    metrics = pd.DataFrame(
        {
            "bugid": all_bugids,
        }
    )

    metrics["plausible"] = (
        metrics["bugid"]
        .map(plausible_by_bug)
        .fillna(False)
        .astype(bool)
    )

    metrics["correct"] = (
        metrics["bugid"]
        .map(correct_by_bug)
        .astype("boolean")
    )

    metrics["identical"] = (
        metrics["bugid"]
        .map(identical_by_bug)
        .fillna(False)
        .astype(bool)
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "MultiMend QuixBugs-Python Validation"
    )
    print("=" * 70)

    print(
        f"Context strategy : {context_strategy}"
    )

    print(
        f"Output directory : {output_dir}"
    )

    print(
        f"Workers          : {N_JOBS}"
    )

    print(
        f"Timeout          : {TEST_TIMEOUT}s"
    )

    # ========================================================
    # LOAD METADATA
    # ========================================================

    bugs_metadata = load_metadata()

    print(
        f"Metadata bugs    : {len(bugs_metadata)}"
    )

    # ========================================================
    # CANDIDATE FILE
    # ========================================================

    candidate_file = (
        output_dir
        / f"final_candidates_{output_size}.jsonl"
    )

    if not candidate_file.is_file():

        raise FileNotFoundError(
            "Candidate file not found:\n"
            f"{candidate_file}"
        )

    candidate_patches_df = pd.read_json(
        candidate_file,
        orient="records",
        lines=True,
    )

    print(
        f"Candidate rows   : "
        f"{len(candidate_patches_df)}"
    )

    # ========================================================
    # RESET STATE
    # ========================================================

    if RESET_VALIDATION:

        print(
            "Resetting validation state..."
        )

        shutil.rmtree(
            save_state_dir,
            ignore_errors=True,
        )

    shutil.rmtree(
        temp_dir,
        ignore_errors=True,
    )

    save_state_dir.mkdir(
        parents=True,
        exist_ok=True,
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
    # VALIDATE
    # ========================================================

    with tqdm_joblib(
        tqdm(
            total=len(bugs_metadata),
            desc=(
                f"Validating "
                f"{context_strategy}"
            ),
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
    # LOAD STATES
    # ========================================================

    state_files = sorted(
        save_state_dir.glob(
            "*.jsonl"
        )
    )

    if not state_files:

        raise RuntimeError(
            "No validation state files were produced."
        )

    cp_dfs = [
        pd.read_json(
            state_file,
            orient="records",
            lines=True,
        )
        for state_file in state_files
    ]

    concatenated_cp_df = pd.concat(
        cp_dfs,
        ignore_index=True,
    )

    # ========================================================
    # CHECK ROW COUNT
    # ========================================================

    if (
        len(candidate_patches_df)
        != len(concatenated_cp_df)
    ):

        raise RuntimeError(
            "Candidate count mismatch: "
            f"input={len(candidate_patches_df)}, "
            f"validated={len(concatenated_cp_df)}"
        )

    # ========================================================
    # GOLD / MANUAL CORRECTNESS
    # ========================================================

    gold_annotations = (
        load_gold_annotations()
    )

    concatenated_cp_df = (
        add_evaluation_labels(
            concatenated_cp_df,
            gold_annotations,
        )
    )

    # ========================================================
    # SAVE CANDIDATE RESULTS
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

    # ========================================================
    # BUG-LEVEL METRICS
    # ========================================================

    bug_metrics = calculate_bug_metrics(
        concatenated_cp_df
    )

    metrics_file = (
        output_dir
        / f"paper_bug_metrics_{output_size}.jsonl"
    )

    bug_metrics.to_json(
        metrics_file,
        orient="records",
        lines=True,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    plausible_count = int(
        bug_metrics[
            "plausible"
        ].sum()
    )

    total_bugs = len(
        bug_metrics
    )

    identical_count = int(
        bug_metrics[
            "identical"
        ].sum()
    )

    print()
    print("=" * 70)
    print(
        f"RESULTS: {context_strategy}"
    )
    print("=" * 70)

    print(
        f"Total bugs       : {total_bugs}"
    )

    print(
        f"Plausible bugs   : "
        f"{plausible_count}"
    )

    print(
        f"Plausible rate   : "
        f"{plausible_count / total_bugs:.4f}"
        if total_bugs
        else "Plausible rate   : N/A"
    )

    print(
        f"Identical bugs   : "
        f"{identical_count}"
    )

    # --------------------------------------------------------
    # GOLD CORRECTNESS
    # --------------------------------------------------------

    if (
        bug_metrics["correct"]
        .notna()
        .any()
    ):

        correct_count = int(
            bug_metrics[
                "correct"
            ]
            .fillna(False)
            .sum()
        )

        correct_total = int(
            bug_metrics[
                "correct"
            ]
            .notna()
            .sum()
        )

        print(
            f"Correct bugs     : "
            f"{correct_count}"
            f" / "
            f"{correct_total}"
        )

        if correct_total:

            print(
                f"Correct rate     : "
                f"{correct_count / correct_total:.4f}"
            )

    else:

        print(
            "Correct bugs     : "
            "N/A (no gold/manual annotations)"
        )

    # --------------------------------------------------------
    # PAPER-STYLE COUNTS
    # --------------------------------------------------------

    print()
    print("Plausible value counts:")
    print(
        bug_metrics[
            "plausible"
        ]
        .value_counts()
    )

    if (
        bug_metrics["correct"]
        .notna()
        .any()
    ):

        print()
        print("Correct value counts:")
        print(
            bug_metrics[
                "correct"
            ]
            .value_counts(
                dropna=False
            )
        )

    print()
    print(
        f"Candidate results -> {result_file}"
    )

    print(
        f"Bug metrics       -> {metrics_file}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()