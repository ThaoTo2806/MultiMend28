import os
from itertools import chain
from pathlib import Path

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

# Config
dataset = "QuixBugs-Python"
model = "multimend"

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


context_strategy = os.environ.get("MULTIMEND_CONTEXT_STRATEGY", "fixed_rag")
if context_strategy not in {"no_rag", "fixed_rag"}:
    raise ValueError(f"Unsupported context strategy: {context_strategy}")
output_dir = gen_dir / f"outputs-{model}-{context_strategy}"

output_size = 100
num_checkpoints = 5
paper_correct_file = os.environ.get("MULTIMEND_PAPER_CORRECT_FILE")

rem_file_path = gen_dir / "rem.txt"
add_file_path = gen_dir / "add.txt"

with (
    open(rem_file_path) as rem_file,
    open(add_file_path) as add_file,
):
    sources = [src.strip() for src in rem_file]
    targets = [tgt.strip() for tgt in add_file]


def add_source_target(df: pd.DataFrame) -> pd.DataFrame:
    rem_file_path = gen_dir / "rem.txt"
    add_file_path = gen_dir / "add.txt"

    with (
        open(rem_file_path) as rem_file,
        open(add_file_path) as add_file,
    ):
        sources = [src.strip() for src in rem_file]
        targets = [tgt.strip() for tgt in add_file]

    checkpoints_num = len(df.value_counts("checkpoint"))
    assert checkpoints_num == num_checkpoints

    df["source"] = list(chain(*[[s] * output_size for s in sources])) * num_checkpoints
    df["target"] = list(chain(*[[t] * output_size for t in targets])) * num_checkpoints

    for bugid, group in df.groupby("bugid"):
        assert len(group["target"].unique()) <= len(group["hunk"].unique())

    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df["decoded_sequences"] = df["decoded_sequences"].str.strip()
    df["normalized_patch"] = df["decoded_sequences"].str.split().str.join(sep=" ")
    df["normalized_source"] = df["source"].str.split().str.join(sep=" ")
    df["normalized_target"] = df["target"].str.split().str.join(sep=" ")
    return df


def create_empty_patch(patch_sample: pd.Series) -> pd.DataFrame:
    patch_sample.loc["decoded_sequences"] = ""
    patch_sample.loc["sequences_scores"] = 0
    patch_sample.loc["normalized_patch"] = ""
    patch_sample.loc["checkpoint"] = "manual"
    patch_sample.loc["rank"] = 0

    return pd.DataFrame([patch_sample])


def combine_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """deduplicate, sort and combine candidate patches of different checkpoints"""

    dfs = []
    for _, subset_df in df.groupby(["bugid", "hunk", "checkpoint"]):
        subset_df["rank"] = subset_df.reset_index(drop=True).index
        dfs.append(subset_df)

    ranked_df = pd.concat(dfs)

    ranked_df.loc[df["normalized_patch"] == "", ["rank", "sequences_scores"]] = [0, 0]

    # Should sort based on scores before deduplication for `keep=first` to take effect
    sorted_df = ranked_df.sort_values(
        by=["bugid", "hunk", "rank", "sequences_scores"],
        ascending=[True, True, True, False],
        inplace=False,
        ignore_index=True,
    )

    src_neq_df = sorted_df.loc[
        sorted_df["normalized_patch"] != sorted_df["normalized_source"]
    ]
    sorted_df = src_neq_df.copy()

    deduped_df = sorted_df.drop_duplicates(
        subset=["bugid", "hunk", "normalized_patch"],
        inplace=False,
        ignore_index=True,
    )

    # Adding empty patch to hunks
    concat_dfs = []
    grouped_df = deduped_df.groupby(["bugid", "hunk"])
    for _, group_df in grouped_df:
        if (
            "" not in group_df["normalized_patch"].values
            and group_df["normalized_source"].values[0]
        ):
            empty_patch = create_empty_patch(group_df.iloc[-1].copy())
            concat_dfs.append(pd.concat([empty_patch, group_df], ignore_index=True))
        else:
            concat_dfs.append(group_df)

    return pd.concat(concat_dfs, ignore_index=True)


def load_paper_correct_annotations() -> dict[str, bool]:
    """Load manually reviewed bug labels used by the paper protocol."""

    if not paper_correct_file:
        return {}

    annotation_path = Path(paper_correct_file)
    if not annotation_path.is_file():
        raise FileNotFoundError(
            "MULTIMEND_PAPER_CORRECT_FILE does not exist: "
            f"{annotation_path}"
        )

    annotations = pd.read_json(
        annotation_path,
        orient="records",
        lines=True,
    )
    required_columns = {"bugid", "paper_correct"}
    missing_columns = required_columns - set(annotations.columns)
    if missing_columns:
        raise ValueError(
            "Manual annotation file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    annotations = annotations[["bugid", "paper_correct"]]
    conflicting = (
        annotations.groupby("bugid")["paper_correct"].nunique() > 1
    )
    if conflicting.any():
        raise ValueError(
            "Manual annotation file has conflicting labels for bugids: "
            f"{sorted(conflicting[conflicting].index.astype(str))}"
        )
    annotations = annotations.drop_duplicates("bugid")
    if annotations["paper_correct"].isna().any():
        raise ValueError("Manual paper_correct labels cannot be null")

    return dict(
        zip(
            annotations["bugid"].astype(str),
            annotations["paper_correct"].astype(bool),
        )
    )


def set_evaluation_labels(
    df: pd.DataFrame,
    paper_correct_annotations: dict[str, bool],
) -> pd.DataFrame:
    # Identical:
    # generated patch matches the developer patch.
    df["identical"] = (
        df["normalized_patch"]
        == df["normalized_target"]
    )

    # Correct:
    # manual/semantic correctness.
    # Do NOT infer correctness from identical.
    df["paper_correct"] = (
        df["bugid"]
        .astype(str)
        .map(paper_correct_annotations)
        .astype("boolean")
    )

    return df


def main():
    checkpoints_results = pd.read_json(
        output_dir / f"sequences_{output_size}.jsonl",
        orient="records",
        lines=True,
    )

    checkpoint_counts = checkpoints_results["checkpoint"].value_counts()
    if len(checkpoint_counts) != num_checkpoints:
        raise RuntimeError(
            f"Expected {num_checkpoints} checkpoints, found "
            f"{len(checkpoint_counts)}: {checkpoint_counts.to_dict()}"
        )
    expected_rows_per_checkpoint = len(sources) * output_size
    invalid_counts = checkpoint_counts[
        checkpoint_counts != expected_rows_per_checkpoint
    ]
    if not invalid_counts.empty:
        raise RuntimeError(
            "Each checkpoint must contain "
            f"{expected_rows_per_checkpoint} rows; invalid counts: "
            f"{invalid_counts.to_dict()}"
        )

    column_index = (
        checkpoints_results.columns[-2:].to_list()
        + checkpoints_results.columns[:-2].to_list()
    )

    checkpoints_results = checkpoints_results[column_index]
    print("All:", len(checkpoints_results))
    add_source_target(checkpoints_results)

    deduped_df = combine_candidates(normalize(checkpoints_results))
    print("Deduped:", len(deduped_df))
    paper_correct_annotations = load_paper_correct_annotations()
    set_evaluation_labels(
    deduped_df,
    paper_correct_annotations,
)

    if paper_correct_annotations:
        missing_bugs = set(deduped_df["bugid"]) - set(
            paper_correct_annotations
        )
        if missing_bugs:
            raise ValueError(
                "Manual annotations are incomplete; missing bugids: "
                f"{sorted(missing_bugs)}"
            )
        print(
            "Paper metric: manual correctness labels loaded for "
            f"{len(paper_correct_annotations)} bugs"
        )
    else:
        print(
            "Paper metric: unavailable until "
            "MULTIMEND_PAPER_CORRECT_FILE is provided"
        )

    deduped_df.to_json(
        output_dir / f"final_candidates_{output_size}.jsonl",
        orient="records",
        lines=True,
    )


if __name__ == "__main__":
    main()
