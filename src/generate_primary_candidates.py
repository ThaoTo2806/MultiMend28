import json
import os
import string
from collections import ChainMap, defaultdict
from itertools import chain
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
)

from .configs import (
    bugaid_gen_dir,
    bugsinpy_gen_dir,
    codeflaws_gen_dir,
    d4j_gen_dir,
    models_root,
    quixbugs_genjava_dir,
    quixbugs_genpy_dir,
    runbugrunjs_gen_dir,
)
from .rag_utils import RAG

set_seed(42)

# Config
dataset = "QuixBugs-Python"
model_name = "multimend-codet5-small"
context_strategy = os.environ.get("MULTIMEND_CONTEXT_STRATEGY", "fixed_rag")
if context_strategy not in {"no_rag", "fixed_rag"}:
    raise ValueError(f"Unsupported context strategy: {context_strategy}")


def get_checkpoints(checkpoints_dir: Path) -> list[tuple[str, Path]]:
    """Return the final checkpoints ordered by trainer_state global_step."""

    checkpoints: list[tuple[int, str, Path]] = []
    for checkpoint in checkpoints_dir.glob("checkpoint-*"):
        state_file = checkpoint / "trainer_state.json"
        if not state_file.is_file():
            continue
        state = json.loads(state_file.read_text(encoding="utf-8"))
        step = state.get("global_step")
        if isinstance(step, int) and step >= 0:
            checkpoints.append((step, checkpoint.name, checkpoint))
    checkpoints.sort(key=lambda item: item[0])
    if len(checkpoints) < num_checkpoints:
        raise RuntimeError(
            f"Need at least {num_checkpoints} valid checkpoints, found {len(checkpoints)}"
        )
    selected_checkpoints = checkpoints[-num_checkpoints:]
    if len({step for step, _, _ in selected_checkpoints}) != num_checkpoints:
        raise RuntimeError("Selected checkpoints have duplicate global_step values")
    print("Selected checkpoints:")
    for step, name, _ in selected_checkpoints:
        print(f"- {name}: global_step={step}")
    return [
        (name, checkpoint)
        for step, name, checkpoint in selected_checkpoints
    ]


def load_test_input_from_meta(prefix: str) -> Dataset:
    """Build inputs and record retrieval/truncation instrumentation."""

    def prepare(hunk: str) -> str:
        lines_concat = " ".join([line.strip() for line in hunk.splitlines()])
        return lines_concat.strip()

    n_return = 5
    threshold = 0.5
    rag = RAG(dataset) if context_strategy == "fixed_rag" else None
    test_data = defaultdict(list)
    context_log: list[dict] = []

    with open(gen_dir / bugs_metadata_file) as meta_file:
        bugs_metadata = ChainMap(*[json.loads(line) for line in meta_file][::-1])

    print("# RAG retrievals...")
    for bugid, hunks in bugs_metadata.items():
        for h, hunk in enumerate(hunks):
            src = prepare(hunk["removed_lines"])
            source = f"{prefix} {src} :"
            context = " ".join(hunk["source_context"][0].split())

            metadata = {"bugid": bugid, "hunk": h}
            docs: list[str] = []
            distances: list[float] = []
            raw_docs: list[str] = []
            raw_distances: list[float] = []
            if rag is not None and src.strip(string.punctuation + string.whitespace):
                raw_docs, _, raw_distances = rag.retrieve_with_scores(
                    src, metadata, n_return
                )
                selected = [
                    (doc, distance)
                    for doc, distance in zip(raw_docs, raw_distances)
                    if distance <= threshold
                ]
                docs = [doc for doc, _ in selected]
                distances = [distance for _, distance in selected]
            rag_result = " ".join(docs)

            print(bugid, h)
            if rag_result:
                test_input = f"{source} {rag_result} {context}".replace(
                    tokenizer.eos_token, tokenizer.unk_token
                )
            else:
                test_input = f"{source} {context}".replace(
                    tokenizer.eos_token, tokenizer.unk_token
                )
            test_data["inputs"].append(test_input)
            untruncated_tokens = len(
                tokenizer(test_input, truncation=False)["input_ids"]
            )
            truncated_tokens = len(
                tokenizer(
                    test_input,
                    truncation=True,
                    max_length=max_input_length,
                )["input_ids"]
            )
            local_only = f"{source} {context}".replace(
                tokenizer.eos_token, tokenizer.unk_token
            )
            local_tokens = len(tokenizer(local_only, truncation=False)["input_ids"])
            context_log.append(
                {
                    "bugid": bugid,
                    "hunk": h,
                    "context_strategy": context_strategy,
                    "retrieved_count": len(docs),
                    "raw_candidates": [
                        {
                            "document": doc,
                            "distance": distance,
                            "similarity": 1 - distance,
                        }
                        for doc, distance in zip(raw_docs, raw_distances)
                    ],
                    "selected_distances": distances,
                    "input_tokens_before_truncation": untruncated_tokens,
                    "input_tokens_after_truncation": truncated_tokens,
                    "local_context_tokens_without_retrieval": local_tokens,
                    "local_context_at_risk": (
                        untruncated_tokens > max_input_length
                        and local_tokens <= max_input_length
                    ),
                    "was_truncated": untruncated_tokens > max_input_length,
                    "threshold": threshold if rag is not None else None,
                }
            )

    test_dataset = Dataset.from_dict(test_data)
    log_path = gen_dir / f"context_log_{context_strategy}.jsonl"
    with log_path.open("w", encoding="utf-8") as log_file:
        for record in context_log:
            log_file.write(json.dumps(record) + "\n")
    return test_dataset


def tokenize_data(examples):
    inputs = tokenizer(
        examples["inputs"],
        padding="longest",
        truncation=True,
        max_length=max_input_length,
    )
    return inputs


def generate_candidates(examples, model, ch_name):
    input_ids = examples["input_ids"].to(device)
    attention_mask = examples["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            num_beams=beam_size,
            early_stopping=True,
            max_length=max_target_length,
            min_length=min_target_length,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
        )

    outputs_str = tokenizer.batch_decode(
        outputs["sequences"],
        skip_special_tokens=True,
        cleanup_tokenization_spaces=False,
    )

    return {
        "checkpoint": [ch_name] * len(outputs_str),
        "decoded_sequences": outputs_str,
        "sequences_scores": outputs["sequences_scores"].cpu().numpy(),
    }


def save_results(checkpoints_results: list[Dataset]) -> None:
    concatenated_results = concatenate_datasets(checkpoints_results)

    with open(gen_dir / bugs_metadata_file) as meta_file:
        bugs_metadata = ChainMap(*[json.loads(line) for line in meta_file][::-1])

    # Create bugid and hunks for a single checkpoint
    input_bugs_hunks = defaultdict(list)
    for bugid, hunks in bugs_metadata.items():
        input_bugs_hunks["bugid"] += [bugid] * (num_return_sequences * len(hunks))
        input_bugs_hunks["hunk"] += chain.from_iterable(
            [i] * num_return_sequences for i in range(len(hunks))
        )

    # Repeat bugid and hunks in the number of checkpoints
    for colname, coldata in input_bugs_hunks.items():
        input_bugs_hunks[colname] = coldata * len(checkpoints_results)

    bugid_added = concatenated_results.add_column("bugid", input_bugs_hunks["bugid"])
    hunk_added = bugid_added.add_column("hunk", input_bugs_hunks["hunk"])

    total_hunks = sum(len(hunks) for hunks in bugs_metadata.values())
    expected_rows = total_hunks * num_return_sequences * len(checkpoints_results)
    if len(hunk_added) != expected_rows:
        raise RuntimeError(
            f"Generation produced {len(hunk_added)} rows; "
            f"expected {expected_rows} "
            f"({total_hunks} hunks x {num_return_sequences} candidates "
            f"x {len(checkpoints_results)} checkpoints)"
        )

    hunk_added.to_json(output_dir / f"sequences_{beam_size}.jsonl")


if dataset == "QuixBugs-Python":
    gen_dir = quixbugs_genpy_dir
    bugs_metadata_file = "QuixBugs_Python.jsonl"
    prefix = "Python"
elif dataset == "QuixBugs-Java":
    gen_dir = quixbugs_genjava_dir
    bugs_metadata_file = "QuixBugs_Java.jsonl"
    prefix = "Java"
elif dataset == "Defects4J":
    gen_dir = d4j_gen_dir
    bugs_metadata_file = "Defects4J.jsonl"
    prefix = "Java"
elif dataset == "BugAID":
    gen_dir = bugaid_gen_dir
    bugs_metadata_file = "BugAID.jsonl"
    prefix = "JavaScript"
elif dataset == "Codeflaws":
    gen_dir = codeflaws_gen_dir
    bugs_metadata_file = "Codeflaws.jsonl"
    prefix = "C"
elif dataset == "BugsInPy":
    gen_dir = bugsinpy_gen_dir
    bugs_metadata_file = "BugsInPy.jsonl"
    prefix = "Python"
elif dataset == "RunBugRun-JS":
    gen_dir = runbugrunjs_gen_dir
    bugs_metadata_file = "RunBugRun-JS.jsonl"
    prefix = "JavaScript"
else:
    raise ValueError("Wrong dataset name")


checkpoints_dir = models_root / model_name
output_dir = gen_dir / f"outputs-{model_name.split('-')[0]}-{context_strategy}"

max_input_length = 512
max_target_length = 256
min_target_length = 0
beam_size = 100
num_return_sequences = 100
batch_size = 1
num_checkpoints = 5

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

checkpoints = get_checkpoints(checkpoints_dir)
if len(checkpoints) != num_checkpoints:
    raise RuntimeError(
        f"Checkpoint selection returned {len(checkpoints)} entries; "
        f"expected exactly {num_checkpoints}"
    )
tokenizer = AutoTokenizer.from_pretrained(checkpoints[0][1])

output_dir.mkdir(parents=True, exist_ok=True)
test_dataset = load_test_input_from_meta(prefix)
test_dataset.to_json(output_dir / "generated_input.jsonl")

print("Tokenizing...")
tokenized_test_dataset = test_dataset.map(
    tokenize_data,
    batched=True,
    remove_columns=test_dataset.column_names,
)
tokenized_test_dataset.set_format("torch")
output_dir.mkdir(exist_ok=True)

checkpoints_results: list[Dataset] = []
for ch_name, checkpoint in checkpoints:
    print(f"Generating from {ch_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    decoder_start_token_id = next(
        (
            token_id
            for token_id in (
                model.config.decoder_start_token_id,
                model.generation_config.decoder_start_token_id,
                tokenizer.pad_token_id,
            )
            if token_id is not None
        ),
        None,
    )
    if decoder_start_token_id is None:
        raise ValueError(
            f"Checkpoint {ch_name} has no decoder_start_token_id and tokenizer "
            "has no pad_token_id"
        )
    model.generation_config.decoder_start_token_id = decoder_start_token_id
    model.generation_config.bos_token_id = model.config.bos_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = next(
        (
            token_id
            for token_id in (model.config.pad_token_id, tokenizer.pad_token_id)
            if token_id is not None
        ),
        None,
    )
    results = tokenized_test_dataset.map(
        lambda examples: generate_candidates(examples, model, ch_name),
        batched=True,
        batch_size=batch_size,
        remove_columns=tokenized_test_dataset.column_names,
    )
    checkpoints_results.append(results)
    torch.cuda.empty_cache()

save_results(checkpoints_results)
