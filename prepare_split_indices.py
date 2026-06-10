from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_DATA_DIR = BASE_DIR / "input_data"

DATASET_PATH = INPUT_DATA_DIR / "MD_dataset.csv"
SPLIT_PATH = INPUT_DATA_DIR / "split_tau_0p800000.tsv"
TRAIN_INDEX_PATH = INPUT_DATA_DIR / "train_indices_md.txt"
VALID_INDEX_PATH = INPUT_DATA_DIR / "validation_indices_md.txt"
TEST_INDEX_PATH = INPUT_DATA_DIR / "test_indices_md.txt"

EXPECTED_SPLITS = {"train", "validation", "test"}


def write_index_file(path, indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{index}\n" for index in indices), encoding="utf-8")


def main():
    dataset = pd.read_csv(DATASET_PATH, usecols=["pdbid", "chain"])
    split = pd.read_csv(SPLIT_PATH, sep="\t", usecols=["pdbid", "chain", "split"])

    if dataset.duplicated(["pdbid", "chain"]).any():
        duplicates = dataset.loc[
            dataset.duplicated(["pdbid", "chain"], keep=False),
            ["pdbid", "chain"],
        ]
        raise ValueError(
            "Dataset contains duplicate pdbid/chain pairs:\n"
            f"{duplicates.head(10).to_string(index=False)}"
        )

    if split.duplicated(["pdbid", "chain"]).any():
        duplicates = split.loc[
            split.duplicated(["pdbid", "chain"], keep=False),
            ["pdbid", "chain"],
        ]
        raise ValueError(
            "Split file contains duplicate pdbid/chain pairs:\n"
            f"{duplicates.head(10).to_string(index=False)}"
        )

    merged = dataset.reset_index().merge(
        split,
        on=["pdbid", "chain"],
        how="left",
        indicator=True,
    )

    missing = merged.loc[merged["_merge"] != "both", ["index", "pdbid", "chain"]]
    if not missing.empty:
        raise ValueError(
            "Missing split assignments for dataset rows:\n"
            f"{missing.head(10).to_string(index=False)}"
        )

    observed_splits = set(merged["split"].dropna().unique())
    unknown_splits = observed_splits - EXPECTED_SPLITS
    if unknown_splits:
        raise ValueError(f"Unexpected split labels found: {sorted(unknown_splits)}")

    train_indices = merged.loc[merged["split"] == "train", "index"].tolist()
    valid_indices = merged.loc[merged["split"] == "validation", "index"].tolist()
    test_indices = merged.loc[merged["split"] == "test", "index"].tolist()

    if not train_indices:
        raise ValueError("No train indices were generated.")

    if not valid_indices:
        raise ValueError("No validation indices were generated.")

    if not test_indices:
        raise ValueError("No test indices were generated.")

    write_index_file(TRAIN_INDEX_PATH, train_indices)
    write_index_file(VALID_INDEX_PATH, valid_indices)
    write_index_file(TEST_INDEX_PATH, test_indices)

    split_counts = merged["split"].value_counts().to_dict()
    print(f"Wrote {len(train_indices)} train indices to {TRAIN_INDEX_PATH}")
    print(f"Wrote {len(valid_indices)} validation indices to {VALID_INDEX_PATH}")
    print(f"Wrote {len(test_indices)} test indices to {TEST_INDEX_PATH}")
    print(f"Split counts from TSV: {split_counts}")


main()
