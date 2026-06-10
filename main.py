#%%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric as tg

from utils.utils_data import load_or_create_split_indices, prepare_dataset
from utils.utils_model import e3net, train

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
valid_idx_path = (INPUT_DATA_DIR / "validation_indices_md.txt").resolve()
test_idx_path = (INPUT_DATA_DIR / "test_indices_md.txt").resolve()

save_graph_cache = False
rebuild_graph_cache = False
r_max = 12.0
run_name = str(BASE_DIR / "model_PC")

#%%
# One-hot encoding for residue names
residues = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
residue_encoding = {residue: index for index, residue in enumerate(residues)}

#%%
if not dataset_path.exists():
    raise FileNotFoundError(f"Place MD_dataset.csv at {dataset_path} before training.")

# Load data and build processed graphs from the CSV.

data_all = prepare_dataset(
    dataset_path=dataset_path,
    graph_cache_path=graph_cache_path,
    residue_encoding=residue_encoding,
    r_max=r_max,
    save_graph_cache=save_graph_cache,
    rebuild_graph_cache=rebuild_graph_cache,
)

#%%
# Load train / valid / test splits. If the small split files are missing,
# create a deterministic 80/10/10 split from the CSV rows.
idx_train, idx_valid, idx_test = load_or_create_split_indices(
    data_all,
    train_idx_path,
    valid_idx_path,
    test_idx_path,
)

max_index = len(data_all) - 1
invalid_train = [index for index in idx_train if index < 0 or index > max_index]
invalid_valid = [index for index in idx_valid if index < 0 or index > max_index]
invalid_test = [index for index in idx_test if index < 0 or index > max_index]

if invalid_train:
    raise ValueError(f"Train indices out of range: {invalid_train[:10]}")
if invalid_valid:
    raise ValueError(f"Validation indices out of range: {invalid_valid[:10]}")
if invalid_test:
    raise ValueError(f"Test indices out of range: {invalid_test[:10]}")

batch_size = 1
data_train = tg.loader.DataLoader(data_all.iloc[idx_train]["data"].tolist(), batch_size=batch_size, shuffle=True)
data_valid = tg.loader.DataLoader(data_all.iloc[idx_valid]["data"].tolist(), batch_size=batch_size)
data_test = tg.loader.DataLoader(data_all.iloc[idx_test]["data"].tolist(), batch_size=1)

#%%
# Define model hyperparameters
name_model = run_name
inter_irreps = "12x0o+12x0e+8x1o+8x1e+8x2o+8x2e"
num_layer = 3
lmax = 3
max_iter = 100
max_radius = r_max
fcn_len = 16
number_of_basis = 16
irreps_pc_vec = "1x1o"
lr = 0.01
scheduler_patience = 10
scheduler_factor = 0.5
weight_ratio = 1.0

#%%
# Build the model
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

#%%
# Train the model
history = train(
    gnn_model,
    lr,
    data_train,
    data_valid,
    loss_fn=torch.nn.MSELoss(),
    loss_fn_mae=torch.nn.L1Loss(),
    run_name=name_model,
    max_iter=max_iter,
    device=device,
    weight_ratio=weight_ratio,
    scheduler_patience=scheduler_patience,
    scheduler_factor=scheduler_factor,
    dataloader_test=data_test,
)

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
        pred_pc = pred_pc / pred_norm

        dot_prod = torch.dot(pred_pc, ref_pc)
        loss1 = abs(1.0 - dot_prod)
        loss2 = abs(1.0 + dot_prod)

        if loss1 < loss2:
            loss = loss1
            adjusted_pred_pc = pred_pc
        else:
            loss = loss2
            adjusted_pred_pc = -pred_pc

        all_loss.append(loss.item())
        all_pred.append(adjusted_pred_pc.cpu().numpy())
        all_ref.append(ref_pc.cpu().numpy())

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
plt.show()

#%%
# Plot and save the convergence of loss over training steps
steps = [entry["step"] for entry in history]
train_losses = [entry["train_loss"] for entry in history]
valid_losses = [entry["valid_loss"] for entry in history]
test_losses = [entry["test_loss"] for entry in history]

plt.figure(figsize=(10, 6))
plt.plot(steps, train_losses, label="Training Loss", marker="o")
plt.plot(steps, valid_losses, label="Validation Loss", marker="s")
plt.plot(steps, test_losses, label="Test Loss", marker="^")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Convergence of Training, Validation, and Test Loss")
plt.legend()
plt.yscale("log")
plt.grid(True)
plt.savefig(BASE_DIR / "loss_convergence_v3.png")
plt.show()
