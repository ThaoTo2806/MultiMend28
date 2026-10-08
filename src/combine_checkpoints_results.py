import os
from itertools import chain

import pandas as pd

from .configs import (
    bugaid_gen_dir,
    bugsinpy_gen_dir,
    codeflaws_gen_dir,
    d4j_gen_dir,
    quixbugs_genjava_dir,
    quixbugs_genpy_dir,
    runbugrunjs_gen_dir,
)

# ============================================================
# Config
# ============================================================

dataset = "QuixBugs-Python"
model = "multimend"

context_strategy = os.environ.get(
    "MULTIMEND_CONTEXT_STRATEGY",
    "fixed_rag",
)

if context_strategy not in {"no_rag", "fixed_rag"}:
    raise ValueError(
        f"Unsupported context strategy: {context_strategy}"
    )


# ============================================================
# Dataset
# ============================================================

if dataset == "QuixBugs-Python":
    gen_dir = quixbugs_genpy_dir
    bugs_metadata_file = "QuixBugs_Python.jsonl"

elif dataset == "QuixBugs-Java":
    gen_dir = quixbugs_genjava_dir
    bugs_metadata_file = "QuixBugs_Java.jsonl"

elif dataset == "Defects4J":
    gen_dir = d4j_gen_dir
    bugs_metadata_file = "Defects4J.jsonl"

elif dataset == "BugAID":
    gen_dir = bugaid_gen_dir
    bugs_metadata_file = "BugAID.jsonl"

elif dataset == "Codeflaws":
    gen_dir = codeflaws_gen_dir
    bugs_metadata_file = "Codeflaws.jsonl"

elif dataset == "BugsInPy":
    gen_dir = bugsinpy_gen_dir
    bugs_metadata_file = "BugsInPy.jsonl"

elif dataset == "RunBugRun-JS":
    gen_dir = runbugrunjs_gen_dir
    bugs_metadata_file = "RunBugRun-JS.jsonl"

else:
    raise ValueError("Wrong dataset name")


output_dir = (
    gen_dir
    / f"outputs-{model}-{context_strategy}"
)

output_size = 100
num_checkpoints = 5


# ============================================================
# Source / target
# ============================================================

rem_file_path = gen_dir / "rem.txt"
add_file_path = gen_dir / "add.txt"

with (
    open(rem_file_path, encoding="utf-8") as rem_file,
    open(add_file_path, encoding="utf-8") as add_file,
):
    sources = [src.strip() for src in rem_file]
    targets = [tgt.strip() for tgt in add_file]


def add_source_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add developer source and target patches.

    The generation order is:

        bug/hunk
        x 100 candidates
        x 5 checkpoints
    """

    checkpoints_num = df["checkpoint"].nunique()

    if checkpoints_num != num_checkpoints:
        raise RuntimeError(
            f"Expected {num_checkpoints} checkpoints, "
            f"found {checkpoints_num}"
        )

    expected_rows = (
        len(sources)
        * output_size
        * num_checkpoints
    )

    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} generated rows, "
            f"found {len(df)}"
        )

    source_values = list(
        chain.from_iterable(
            [[source] * output_size for source in sources]
        )
    )

    target_values = list(
        chain.from_iterable(
            [[target] * output_size for target in targets]
        )
    )

    df["source"] = (
        source_values
        * num_checkpoints
    )

    df["target"] = (
        target_values
        * num_checkpoints
    )

    return df


# ============================================================
# Normalize
# ============================================================

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize generated patches and developer patches
    for textual comparison.
    """

    df = df.copy()

    df["decoded_sequences"] = (
        df["decoded_sequences"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["normalized_patch"] = (
        df["decoded_sequences"]
        .str.split()
        .str.join(" ")
    )

    df["normalized_source"] = (
        df["source"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.join(" ")
    )

    df["normalized_target"] = (
        df["target"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.join(" ")
    )

    return df


# ============================================================
# Empty patch
# ============================================================

def create_empty_patch(
    patch_sample: pd.Series,
) -> pd.DataFrame:

    patch_sample = patch_sample.copy()

    patch_sample["decoded_sequences"] = ""
    patch_sample["sequences_scores"] = 0
    patch_sample["normalized_patch"] = ""
    patch_sample["checkpoint"] = "manual"
    patch_sample["rank"] = 0

    return pd.DataFrame([patch_sample])


# ============================================================
# Combine candidates
# ============================================================

def combine_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine candidates from the five checkpoints.

    Paper-style processing:

    1. Rank candidates inside each checkpoint.
    2. Sort by rank and model score.
    3. Remove candidates identical to source.
    4. Deduplicate candidates across checkpoints.
    5. Add an empty patch candidate.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Rank candidates inside every checkpoint
    # --------------------------------------------------------

    ranked_groups = []

    for (_, _, _), group in df.groupby(
        ["bugid", "hunk", "checkpoint"],
        sort=False,
    ):
        group = group.copy()

        group["rank"] = (
            group
            .reset_index(drop=True)
            .index
        )

        ranked_groups.append(group)

    if not ranked_groups:
        raise RuntimeError(
            "No candidate patches found."
        )

    ranked_df = pd.concat(
        ranked_groups,
        ignore_index=True,
    )

    # Empty generated candidate gets rank 0.
    empty_mask = (
        ranked_df["normalized_patch"] == ""
    )

    ranked_df.loc[
        empty_mask,
        ["rank", "sequences_scores"],
    ] = [0, 0]

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    sorted_df = ranked_df.sort_values(
        by=[
            "bugid",
            "hunk",
            "rank",
            "sequences_scores",
        ],
        ascending=[
            True,
            True,
            True,
            False,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Remove candidate equal to original source
    # --------------------------------------------------------

    sorted_df = sorted_df.loc[
        sorted_df["normalized_patch"]
        != sorted_df["normalized_source"]
    ].copy()

    # --------------------------------------------------------
    # Deduplicate generated patches
    # --------------------------------------------------------

    deduped_df = sorted_df.drop_duplicates(
        subset=[
            "bugid",
            "hunk",
            "normalized_patch",
        ],
        keep="first",
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Add empty patch candidate
    # --------------------------------------------------------

    output_groups = []

    for (_, _), group_df in deduped_df.groupby(
        ["bugid", "hunk"],
        sort=False,
    ):

        group_df = group_df.copy()

        has_empty_patch = (
            ""
            in group_df["normalized_patch"].values
        )

        has_source = bool(
            group_df["normalized_source"]
            .iloc[0]
        )

        if (
            not has_empty_patch
            and has_source
        ):
            empty_patch = create_empty_patch(
                group_df.iloc[-1]
            )

            group_df = pd.concat(
                [
                    empty_patch,
                    group_df,
                ],
                ignore_index=True,
            )

        output_groups.append(group_df)

    if not output_groups:
        raise RuntimeError(
            "No candidates remain after deduplication."
        )

    return pd.concat(
        output_groups,
        ignore_index=True,
    )


# ============================================================
# Main
# ============================================================

def main():

    sequence_file = (
        output_dir
        / f"sequences_{output_size}.jsonl"
    )

    final_file = (
        output_dir
        / f"final_candidates_{output_size}.jsonl"
    )

    if not sequence_file.is_file():
        raise FileNotFoundError(
            f"Generation result not found:\n"
            f"{sequence_file}"
        )

    print(
        f"Loading:\n{sequence_file}"
    )

    checkpoints_results = pd.read_json(
        sequence_file,
        orient="records",
        lines=True,
    )

    # --------------------------------------------------------
    # Validate checkpoints
    # --------------------------------------------------------

    checkpoint_counts = (
        checkpoints_results["checkpoint"]
        .value_counts()
    )

    print("\nCheckpoint counts:")
    print(checkpoint_counts)

    if len(checkpoint_counts) != num_checkpoints:
        raise RuntimeError(
            f"Expected {num_checkpoints} checkpoints, "
            f"found {len(checkpoint_counts)}"
        )

    expected_rows_per_checkpoint = (
        len(sources) * output_size
    )

    invalid_counts = checkpoint_counts[
        checkpoint_counts
        != expected_rows_per_checkpoint
    ]

    if not invalid_counts.empty:
        raise RuntimeError(
            "Invalid number of rows per checkpoint:\n"
            f"{invalid_counts.to_dict()}"
        )

    # --------------------------------------------------------
    # Add source / target
    # --------------------------------------------------------

    print(
        f"\nAll generated candidates: "
        f"{len(checkpoints_results)}"
    )

    checkpoints_results = add_source_target(
        checkpoints_results
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    checkpoints_results = normalize(
        checkpoints_results
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    deduped_df = combine_candidates(
        checkpoints_results
    )

    print(
        f"Deduped candidates: "
        f"{len(deduped_df)}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT calculate "correct" here.
    #
    # Correctness is determined by the validation/reference
    # stage, not simply by normalized_patch == normalized_target.
    # --------------------------------------------------------

    if "correct" in deduped_df.columns:
        deduped_df = deduped_df.drop(
            columns=["correct"]
        )

    if "plausible" in deduped_df.columns:
        deduped_df = deduped_df.drop(
            columns=["plausible"]
        )

    if "paper_correct" in deduped_df.columns:
        deduped_df = deduped_df.drop(
            columns=["paper_correct"]
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    deduped_df.to_json(
        final_file,
        orient="records",
        lines=True,
    )

    print(
        f"\nSaved:\n{final_file}"
    )


if __name__ == "__main__":
    main()