from pathlib import Path
import base64
import zlib

import numpy as np
import pandas as pd
from ase import Atoms
from ase.neighborlist import neighbor_list
import torch
import torch_geometric as tg

#%%
# Default dtype
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


#%%
def detect_schema_tag(columns):
    columns = set(columns)
    if {"pdbid", "chain", "PC", "resid_chain"}.issubset(columns):
        return "nmr"
    if {"pdb_code", "UniProt_ID", "PCs", "nPC"}.issubset(columns):
        return "old"
    raise ValueError(f"Unsupported dataset columns: {sorted(columns)}")


#%%
def parse_numeric_vector(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)

    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64)

    value = decode_compact_value(value)
    return np.fromstring(str(value), sep=",", dtype=np.float64)


#%%
def decode_compact_value(value):
    if not isinstance(value, str):
        return value

    if value.startswith("z85:"):
        compressed = base64.b85decode(value[4:].encode("ascii"))
        return zlib.decompress(compressed).decode("ascii")

    if value.startswith("z64:"):
        compressed = base64.b64decode(value[4:].encode("ascii"))
        return zlib.decompress(compressed).decode("ascii")

    return value


#%%
def load_or_create_split_indices(data_all, train_idx_path, valid_idx_path, test_idx_path):
    split_paths = [train_idx_path, valid_idx_path, test_idx_path]

    if all(path.exists() for path in split_paths):
        indices = []
        for path in split_paths:
            with open(path, "r") as handle:
                indices.append([int(line.strip()) for line in handle if line.strip()])
        return indices

    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(data_all))
    train_end = int(len(shuffled) * 0.8)
    valid_end = int(len(shuffled) * 0.9)

    idx_train = sorted(shuffled[:train_end].tolist())
    idx_valid = sorted(shuffled[train_end:valid_end].tolist())
    idx_test = sorted(shuffled[valid_end:].tolist())

    for path, indices in zip(split_paths, [idx_train, idx_valid, idx_test]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{index}\n" for index in indices), encoding="utf-8")

    return idx_train, idx_valid, idx_test


#%%
def graph_build(data, residue_encoding, r_max):
    schema_tag = detect_schema_tag(data.index)
    residues = [residue for residue in decode_compact_value(data["resid_type"]).split(",") if residue]
    num_residues = len(residues)

    coordinate = parse_numeric_vector(data["coordinate"])
    if coordinate.size != num_residues * 3:
        raise ValueError(f"Coordinate length mismatch for row: expected {num_residues * 3}, got {coordinate.size}")
    positions_np = coordinate.reshape(num_residues, 3)

    if schema_tag == "nmr":
        pc_values = parse_numeric_vector(data["PC"])
        chain_labels = list(str(decode_compact_value(data["resid_chain"])))
        var_pc1 = float(data["var"])
    else:
        pc_values = parse_numeric_vector(data["PCs"])[: num_residues * 3]
        chain_label = "A"
        if "pdb_code" in data and "_" in str(data["pdb_code"]):
            chain_label = str(data["pdb_code"]).split("_")[-1]
        chain_labels = [chain_label] * num_residues
        var_pc1 = float(parse_numeric_vector(data["var"])[0])

    if pc_values.size != num_residues * 3:
        raise ValueError(f"PC length mismatch for row: expected {num_residues * 3}, got {pc_values.size}")

    if len(chain_labels) != num_residues:
        raise ValueError(f"Chain label length mismatch for row: expected {num_residues}, got {len(chain_labels)}")

    temp_atoms = Atoms(numbers=[6] * num_residues, positions=positions_np, pbc=[False, False, False])
    edge_src, edge_dst, _ = neighbor_list("ijS", a=temp_atoms, cutoff=r_max, self_interaction=False)

    edge_fea = []
    for src, dst in zip(edge_src, edge_dst):
        same_chain = chain_labels[int(src)] == chain_labels[int(dst)]
        sequential = abs(int(src) - int(dst)) == 1
        edge_fea.append(1.0 if same_chain and sequential else 0.0)

    x = []
    for residue in residues:
        onehot = np.zeros(len(residue_encoding), dtype=np.float64)
        onehot[residue_encoding[residue]] = 1.0
        x.append(onehot)

    data_graph = tg.data.Data(
        x=torch.from_numpy(np.asarray(x, dtype=np.float64)),
        edge_fea=torch.tensor(edge_fea, dtype=default_dtype).unsqueeze(1),
        pos=torch.from_numpy(positions_np).to(default_dtype),
        edge_index=torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0),
        pcs=torch.from_numpy(pc_values.reshape(num_residues * 3, 1)).to(default_dtype),
        var=torch.tensor([var_pc1], dtype=default_dtype),
    )

    return data_graph


#%%
def prepare_dataset(
    dataset_path,
    graph_cache_path,
    residue_encoding,
    r_max,
    save_graph_cache=True,
    rebuild_graph_cache=False,
):
    dataset_path = Path(dataset_path).resolve()
    graph_cache_path = Path(graph_cache_path).resolve()

    if dataset_path.suffix in {".pkl", ".pickle"}:
        data_all = pd.read_pickle(dataset_path)
    elif dataset_path.suffix == ".csv":
        data_all = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset file type: {dataset_path.suffix}")

    schema_tag = detect_schema_tag(data_all.columns)

    metadata = {
        "dataset_path": str(dataset_path),
        "r_max": float(r_max),
        "num_rows": int(len(data_all)),
        "schema_tag": schema_tag,
    }

    if not rebuild_graph_cache and graph_cache_path.exists():
        cache_payload = pd.read_pickle(graph_cache_path)
        cache_metadata = cache_payload.get("metadata", {})
        if cache_metadata == metadata:
            return cache_payload["dataframe"]

    data_all = data_all.copy()
    data_all["data"] = data_all.apply(lambda row: graph_build(row, residue_encoding, r_max), axis=1)

    if save_graph_cache:
        graph_cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"metadata": metadata, "dataframe": data_all}, graph_cache_path)

    return data_all
