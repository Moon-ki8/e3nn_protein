# MD 0p8 training

This directory is a GitHub-ready copy of `MD/0p8` without large data, cache,
checkpoint, or output files.

## Data

Place the dataset at:

```text
input_data/MD_dataset.csv
```

The large graph cache is not required. `main.py` builds graphs directly from
`MD_dataset.csv`. If split index files are missing, the script creates a
deterministic 80/10/10 train/validation/test split.

Optional exact split files can also be placed in `input_data/`:

```text
train_indices_md.txt
validation_indices_md.txt
test_indices_md.txt
```

## Run

```bash
python main.py
```

On the cluster, submit from this directory:

```bash
sbatch run.sh
```

Generated caches, checkpoints, logs, and plots are ignored by `.gitignore`.
