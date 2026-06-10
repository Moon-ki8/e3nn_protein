from pathlib import Path
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric as tg

from utils.utils_data import load_or_create_split_indices, prepare_dataset
from utils.utils_model import e3net


#%%
# Default settings
warnings.filterwarnings("ignore")
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent
INPUT_DATA_DIR = BASE_DIR / "input_data"


#%%
# Dataset / cache / split configuration
dataset_path = (INPUT_DATA_DIR / "MD_dataset.csv").resolve()
graph_cache_path = (INPUT_DATA_DIR / "md_input_dataset_with_graphs.pkl").resolve()
train_idx_path = (INPUT_DATA_DIR / "train_indices_md.txt").resolve()
test_idx_path = (INPUT_DATA_DIR / "test_indices_md.txt").resolve()
model_path = (BASE_DIR / "model_PC.torch").resolve()

r_max = 12.0
batch_size = 1


#%%
# One-hot encoding for residue names
residues = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
residue_encoding = {residue: index for index, residue in enumerate(residues)}


#%%
# Load data and processed graphs
if not dataset_path.exists():
    raise FileNotFoundError(f"Place MD_dataset.csv at {dataset_path} before evaluation.")

data_all = prepare_dataset(
    dataset_path=dataset_path,
    graph_cache_path=graph_cache_path,
    residue_encoding=residue_encoding,
    r_max=r_max,
    save_graph_cache=False,
    rebuild_graph_cache=False,
)


#%%
# Load train / test splits and validate indices
idx_train, _, idx_test = load_or_create_split_indices(
    data_all,
    train_idx_path,
    INPUT_DATA_DIR / "validation_indices_md.txt",
    test_idx_path,
)

max_index = len(data_all) - 1
invalid_train = [index for index in idx_train if index < 0 or index > max_index]
invalid_test = [index for index in idx_test if index < 0 or index > max_index]

if invalid_train:
    raise ValueError(f"Train indices out of range: {invalid_train[:10]}")

if invalid_test:
    raise ValueError(f"Test indices out of range: {invalid_test[:10]}")

data_train = tg.loader.DataLoader(data_all.iloc[idx_train]["data"].tolist(), batch_size=batch_size, shuffle=False)
data_test = tg.loader.DataLoader(data_all.iloc[idx_test]["data"].tolist(), batch_size=1, shuffle=False)

print(f"Loaded {len(data_train.dataset)} train samples and {len(data_test.dataset)} test samples.")


#%%
# Define model hyperparameters
inter_irreps = "12x0o+12x0e+8x1o+8x1e+8x2o+8x2e"
num_layer = 3
lmax = 3
max_radius = r_max
fcn_len = 16
number_of_basis = 16
irreps_pc_vec = "1x1o"


#%%
# Build the model and load checkpoint
gnn_model = e3net(
    in_dim=20,
    irreps_intermediate=inter_irreps,
    irreps_pc_vec=irreps_pc_vec,
    lmax=lmax,
    fcn_len=fcn_len,
    num_layer=num_layer,
    max_radius=max_radius,
    number_of_basis=number_of_basis,
)
gnn_model.to(device)
print(gnn_model)

checkpoint = torch.load(model_path, map_location=torch.device(device))
gnn_model.load_state_dict(checkpoint["state"])


#%%
# Evaluate on the test set
gnn_model.eval()
all_pred = []
all_ref = []
all_loss = []

with torch.no_grad():
    for data in data_test:
        data = data.to(device)

        pred_pc = gnn_model(data).view(-1)
        ref_pc = data.pcs[:, 0].view(-1)

        pred_norm = torch.norm(pred_pc)
        if pred_norm == 0:
            raise ValueError("Predicted principal component has zero norm.")
        pred_pc = pred_pc / pred_norm

        ref_norm_sq = (ref_pc ** 2).sum()
        dot_prod = torch.dot(pred_pc, ref_pc)

        loss1 = abs(ref_norm_sq - dot_prod)
        loss2 = abs(ref_norm_sq + dot_prod)

        if loss1 < loss2:
            loss = loss1
            adjusted_pred_pc = pred_pc
        else:
            loss = loss2
            adjusted_pred_pc = -pred_pc

        all_loss.append(loss.item())
        all_pred.append(adjusted_pred_pc.cpu().numpy())
        all_ref.append(ref_pc.cpu().numpy())

        print(f"loss {loss.item():.6f}")


#%%
# Save raw predictions and references
with open(BASE_DIR / "all_pred.pkl", "wb") as handle:
    pickle.dump(all_pred, handle)

with open(BASE_DIR / "all_ref.pkl", "wb") as handle:
    pickle.dump(all_ref, handle)

print("Saved all_pred.pkl and all_ref.pkl")

all_pred = np.concatenate(all_pred, axis=0)
all_ref = np.concatenate(all_ref, axis=0)

avg_loss = np.mean(all_loss)
print(f"Average evaluation loss: {avg_loss}")


#%%
# Plot predicted vs reference PCs
plt.figure(figsize=(8, 8))
plt.scatter(all_ref, all_pred, alpha=0.6, edgecolor="k", s=30)
plt.xlabel("Reference PC components")
plt.ylabel("Predicted PC components")
plt.title("Predicted vs. Reference Principal Components")
lims = [min(np.min(all_ref), np.min(all_pred)), max(np.max(all_ref), np.max(all_pred))]
plt.plot(lims, lims, "r--", label="Ideal")
plt.legend()
plt.grid(True)
plt.savefig(BASE_DIR / "comparison.png")
