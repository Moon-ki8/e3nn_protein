
import torch
import numpy as np  
from torch_scatter import scatter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, Linear, Irreps
from e3nn.math import soft_one_hot_linspace

import matplotlib.pyplot as plt
import math 
import time 
from tqdm import tqdm 


# format progress bar
bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
###########################################################################################################
def plus2times(x):
    # Concatenate the features to form the input in 0⊕1⊕2
    #x = torch.cat([x0, x1, x2], dim=0)  # Shape: (batch_size, 9)
    # Initialize the tensor product y of shape (batch_size, 3, 3)
    y = torch.zeros(3, 3, device=x.device, dtype=x.dtype)

    # Compute the Wigner 3j symbols and include the (2l + 1) factor
    l1 = l2 = 1  # For 1⊗1
    wigner_3j_coeffs = {}
    for l3 in [0, 1, 2]:
        # Wigner 3j symbols for l1, l2, l3
        coeff = o3.wigner_3j(l1, l2, l3)  # Might be a NumPy array or CPU tensor
        coeff *= 2 * l3 + 1  # Correct the normalization

        # Convert coeff to a torch tensor on the same device and with the same dtype as y
        coeff = torch.tensor(coeff, dtype=y.dtype, device=y.device)
        wigner_3j_coeffs[l3] = coeff

    # For each l, accumulate the contributions to y
    for l3, coeff in wigner_3j_coeffs.items():
        # Retrieve the corresponding features x_l
        if l3 == 0:
            x_lx = x[0]
        elif l3 == 1:
            x_lx = x[1:4]
        elif l3 == 2:
            x_lx = x[4:9]
        else:
            continue

        # Reshape x_l to align with coeff
        x_lx = x_lx.view(1, 1, 2 * l3 + 1) 
        coeff = coeff.view(3, 3, 2 * l3 + 1)    

        # Multiply and sum over the last dimension to get the contribution to y
        contrib = torch.sum(coeff * x_lx, dim=-1)  # Shape: (batch_size, 3, 3)
        y += contrib  # Accumulate the contributions
    return y
###########################################################################################################
def times2plus(y):
    # Initialize dictionaries to hold the Wigner 3j coefficients and output features
    wigner_3j_coeffs = {}
    x_dict = {}

    l1 = l2 = 1  # For 1⊗1
    y = y.view(3, 3)  # Ensure y is of shape (3, 3)

    # Compute the Wigner 3j symbols and include the (2l + 1) factor
    for l3 in [0, 1, 2]:
        # Wigner 3j symbols for l1, l2, l3
        coeff = o3.wigner_3j(l1, l2, l3)  # Shape: (2l1+1, 2l2+1, 2l3+1)
        coeff *= 2 * l3 + 1  # Correct the normalization
        wigner_3j_coeffs[l3] = coeff

        # Compute x_l by projecting y onto the irreducible representation of degree l3
        # x_l[m3] = sum_{m1,m2} y_{m1,m2} * coeff_{m1,m2,m3}
        x_l = torch.einsum('ij,ijm->m', y, coeff)  # Shape: (2l3 + 1,)
        x_dict[l3] = x_l

    # Retrieve x0, x1, x2 from the dictionary
    x0 = x_dict[0]  # Shape: (1,)
    x1 = x_dict[1]  # Shape: (3,)
    x2 = x_dict[2]  # Shape: (5,)

    return x0, x1, x2

##############################################################################
# === Postprocessing utilities for Hamiltonian ===
def compute_center_of_mass(positions, masses):
    total_mass = masses.sum()
    return (positions * masses.unsqueeze(1)).sum(dim=0) / total_mass

def rotation_vectors(positions, masses):
    # positions: (N,3) with COM shifted to zero.
    COM = compute_center_of_mass(positions, masses)
    xyz = positions - COM  # shift to COM = 0
    N = xyz.shape[0]
    r_x = []
    r_y = []
    r_z = []
    for i in range(N):
        x, y, z = xyz[i]
        # Rotation about x-axis: (0, -z, y)
        r_x.append(torch.tensor([0.0, -z, y], dtype=xyz.dtype, device=xyz.device))
        # Rotation about y-axis: (z, 0, -x)
        r_y.append(torch.tensor([z, 0.0, -x], dtype=xyz.dtype, device=xyz.device))
        # Rotation about z-axis: (-y, x, 0)
        r_z.append(torch.tensor([-y, x, 0.0], dtype=xyz.dtype, device=xyz.device))
    return torch.cat(r_x), torch.cat(r_y), torch.cat(r_z)
# === Modified assemble_full_H ===
def assemble_full_H(edge_src, edge_dst, H_elements, num_atoms):
    """
    Assemble the full 3N x 3N Hamiltonian matrix H from the predicted H_elements.
    H_elements is assumed to be a tensor with one entry per edge.
    Each H_element (a vector) is first mapped to a 3x3 block via the function plus2times.
    Then, off-diagonal blocks are symmetrized and the diagonal block is defined as
      H_{ii} = - sum_{j != i} H_{ij}
    Finally, we define H_mult = Hᵀ H to guarantee semipositivity.
    """
    dof = 3 * num_atoms
    H = torch.zeros((dof, dof), dtype=H_elements.dtype, device=H_elements.device)
    # Loop over edges (assumed symmetric)
    for idx in range(edge_src.shape[0]):
        i = int(edge_src[idx])
        j = int(edge_dst[idx])
        i_block = slice(3 * i, 3 * i + 3)
        j_block = slice(3 * j, 3 * j + 3)
        H_block = plus2times(H_elements[idx])
        # Symmetrize the block
        H_block = (H_block + H_block.T) / 2.0
        H[i_block, j_block] += H_block
        H[j_block, i_block] += H_block  # enforce symmetry
    # Set diagonal blocks to negative row-sum (to mimic e.g. a Laplacian-like structure)
    for i in range(num_atoms):
        i_block = slice(3 * i, 3 * i + 3)
        row_sum = torch.zeros((3, 3), dtype=H.dtype, device=H.device)
        for j in range(num_atoms):
            if i == j:
                continue
            j_block = slice(3 * j, 3 * j + 3)
            row_sum += H[i_block, j_block]
        H[i_block, i_block] = -row_sum
    # Build semipositive Hamiltonian H_mult = Hᵀ H
    H_mult = torch.matmul(H.T, H)
    return H, H_mult
# ===P roject_out_rigid_modes ===
def project_out_rigid_modes(H, positions):
    """
    Given a Hamiltonian H (size 3N x 3N) and positions, project out the 6 rigid-body modes.
    We first compute the 6 rotation/translation vectors, orthonormalize them, build
    the projector P, and then form H_new = (I-P) H (I-P).
    """
    N = positions.shape[0]
    dof = 3 * N
    masses = torch.ones(N, dtype=positions.dtype, device=positions.device)  # assume unit masses
    # Translation modes: three vectors (each is the stacking of [1,0,0] etc.)
    trans_x = torch.cat([torch.tensor([1.0,0.0,0.0], dtype=positions.dtype, device=positions.device) for _ in range(N)])
    trans_y = torch.cat([torch.tensor([0.0,1.0,0.0], dtype=positions.dtype, device=positions.device) for _ in range(N)])
    trans_z = torch.cat([torch.tensor([0.0,0.0,1.0], dtype=positions.dtype, device=positions.device) for _ in range(N)])
    # Rotation modes:
    r_x, r_y, r_z = rotation_vectors(positions, masses)
    # Collect all six vectors
    vecs = [trans_x, trans_y, trans_z, r_x, r_y, r_z]
    # Orthonormalize using Gram-Schmidt
    orthonormal = []
    for v in vecs:
        for u in orthonormal:
            v = v - (v @ u) * u
        v_norm = v.norm() + 1e-14
        orthonormal.append(v / v_norm)
    # Build projector P = sum_i v_i v_i^T
    P = torch.zeros((dof, dof), dtype=H.dtype, device=H.device)
    for v in orthonormal:
        P = P + torch.ger(v, v)
    I = torch.eye(dof, dtype=H.dtype, device=H.device)
    H_new = (I - P) @ H @ (I - P)
    return H_new
############################################################################
# Directly imported from e3modules.py in https://github.com/Xiaoxun-Gong/DeepH-E3.git
class e3LayerNorm(nn.Module):
    def __init__(self, irreps_in, eps=1e-5, affine=True, normalization='component', subtract_mean=True, divide_norm=False):
        super().__init__()
        
        self.irreps_in = Irreps(irreps_in)
        self.eps = eps
        
        if affine:          
            ib, iw = 0, 0
            weight_slices, bias_slices = [], []
            for mul, ir in irreps_in:
                if ir.is_scalar(): # bias only to 0e
                    bias_slices.append(slice(ib, ib + mul))
                    ib += mul
                else:
                    bias_slices.append(None)
                weight_slices.append(slice(iw, iw + mul))
                iw += mul
            self.weight = nn.Parameter(torch.ones([iw]))
            self.bias = nn.Parameter(torch.zeros([ib]))
            self.bias_slices = bias_slices
            self.weight_slices = weight_slices
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.subtract_mean = subtract_mean
        self.divide_norm = divide_norm
        assert normalization in ['component', 'norm']
        self.normalization = normalization
            
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.weight is not None:
            self.weight.data.fill_(1)
            # nn.init.uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0)
            # nn.init.uniform_(self.bias)

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None):
        # input x must have shape [num_node(edge), dim]
        # if first dimension of x is node index, then batch should be batch.batch
        # if first dimension of x is edge index, then batch should be batch.batch[batch.edge_index[0]]
        if batch is None:
            batch = torch.full([x.shape[0]], 0, dtype=torch.int64).to(x.device)
        # from torch_geometric.nn.norm.LayerNorm

        batch_size = int(batch.max()) + 1 
        batch_degree = degree(batch, batch_size, dtype=torch.int64).clamp_(min=1).to(dtype=x.dtype)
        
        out = []
        ix = 0
        for index, (mul, ir) in enumerate(self.irreps_in):        
            field = x[:, ix: ix + mul * ir.dim].reshape(-1, mul, ir.dim) # [node, mul, repr]
            
            # compute and subtract mean
            if self.subtract_mean or ir.l == 0: # do not subtract mean for l>0 irreps if subtract_mean=False
                mean = scatter(field, batch, dim=0, dim_size=batch_size,
                            reduce='add').mean(dim=1, keepdim=True) / batch_degree[:, None, None] # scatter_mean does not support complex number
                field = field - mean[batch]
                
            # compute and divide norm
            if self.divide_norm or ir.l == 0: # do not divide norm for l>0 irreps if subtract_mean=False
                norm = scatter(field.abs().pow(2), batch, dim=0, dim_size=batch_size,
                            reduce='mean').mean(dim=[1,2], keepdim=True) # add abs here to deal with complex numbers
                if self.normalization == 'norm':
                    norm = norm * ir.dim
                field = field / (norm.sqrt()[batch] + self.eps)
            
            # affine
            if self.weight is not None:
                weight = self.weight[self.weight_slices[index]]
                field = field * weight[None, :, None]
            if self.bias is not None and ir.is_scalar():
                bias = self.bias[self.bias_slices[index]]
                field = field + bias[None, :, None]
            
            out.append(field.reshape(-1, mul * ir.dim))
            ix += mul * ir.dim
            
        out = torch.cat(out, dim=-1)
                
        return out
    
# E3convolution layer
class E3conv(torch.nn.Module):
    def __init__(self,
                 irreps_node_in,
                 irreps_edge_fea,
                 irreps_out,
                 irreps_edge_attr,
                 fcn_len) -> None: 
        super().__init__()
        self.irreps_node_in = Irreps(irreps_node_in)
        self.irreps_edge_fea = Irreps(irreps_edge_fea)
        self.irreps_out = Irreps(irreps_out) # Need? may be removed
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_intermediate = Irreps(irreps_out)

        self.irreps_in = self.irreps_node_in  + self.irreps_node_in  + self.irreps_edge_fea 
        
        act = {1:torch.nn.functional.silu, -1:torch.tanh}
        act_gates = {1:torch.sigmoid, -1:torch.tanh}

        irreps_gated = Irreps((mul,ir) for mul,ir in self.irreps_intermediate if ir.l > 0)
        num_gated = int(irreps_gated.num_irreps)
        irreps_gates = Irreps(f'{int(num_gated/2)}x0o+{int(num_gated/2)}x0e')
        irreps_scalars = Irreps((mul,ir) for mul,ir in self.irreps_intermediate if ir.l == 0)
        
        

        self.gate = Gate(
            irreps_scalars, [act[ir.p] for _,ir in irreps_scalars], # scalar
            irreps_gates, [act_gates[ir.p] for _,ir in irreps_gates], # gates (scalars)
            irreps_gated,
        )
        
        self.fctp = FullyConnectedTensorProduct(
                self.irreps_in,
                self.irreps_edge_attr,
                self.gate.irreps_in,
                shared_weights=False,
            )
        
        self.fcn = FullyConnectedNet([fcn_len,fcn_len*2,self.fctp.weight_numel],torch.nn.functional.silu)
        
        
        
        self.lin = Linear(irreps_in=self.fctp.irreps_out,irreps_out=self.fctp.irreps_out,biases=False)

    def forward(self, node_in_i, node_in_j,edge_fea,edge_attr,edge_len_embedded):
        weight = self.fcn(edge_len_embedded)
        x = torch.cat((node_in_i, node_in_j, edge_fea), dim=-1)
        x = self.fctp(x,edge_attr,weight)  
        x = self.lin(x)
        x = self.gate(x)
        return x 

# NoteUpdate block
class NodeUpdate(torch.nn.Module):
    def __init__(self,
                 irreps_node_in,
                 irreps_edge_fea,
                 irreps_out,
                 irreps_edge_attr,
                 fcn_len, 
                 ) -> None:
        super().__init__()    
        self.irreps_node_in = irreps_node_in
        self.irreps_edge_fea = irreps_edge_fea
        self.irreps_out = irreps_out
        self.irreps_edge_attr = irreps_edge_attr
        self.fcn_len = fcn_len

        # E3conv 
        self.e3conv = E3conv(
                 self.irreps_node_in,
                 self.irreps_edge_fea,
                 self.irreps_out,
                 self.irreps_edge_attr,
                 self.fcn_len
                 )

        # Linear 
        self.lin = Linear(irreps_in=self.irreps_out,irreps_out=self.irreps_out,biases=False)

        # E3Layernorm 
        self.norm = e3LayerNorm(self.lin.irreps_out)

    def forward(self, node_fea, edge_src, edge_dst, edge_fea, edge_attr, edge_len_embedded):
        # E3conv
        edge_update = self.e3conv(node_fea[edge_src], node_fea[edge_dst], edge_fea, edge_attr, edge_len_embedded)
        # Scatter
        x = scatter(edge_update, edge_src, dim=0, dim_size = node_fea.shape[0], reduce='sum')
        # Linear
        x = self.lin(x)
        # Norm 
        x = self.norm(x)
        node_fea = x 
        return x

    
# E3net model
class e3net(torch.nn.Module):
    def __init__(self,
                in_dim,
                irreps_intermediate,
                irreps_pc_vec,
                lmax,
                fcn_len,
                num_layer,
                max_radius,
                number_of_basis,
                ):
        super().__init__()
        self.irreps_node_in =  Irreps(f'{int(number_of_basis/2)}x0o+{int(number_of_basis/2)}x0e')
        self.irreps_edge_in = Irreps('1x0e')
        self.irreps_intermediate = Irreps(irreps_intermediate)
        self.irreps_edge_attr = Irreps.spherical_harmonics(lmax)
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.fcn_len = fcn_len

        self.irreps_pc_vec = Irreps(irreps_pc_vec)
        self.em = torch.nn.Linear(in_dim, number_of_basis)
        # Layer
        self.node_updates=torch.nn.ModuleList([])
        self.edge_updates=torch.nn.ModuleList([])
        for _ in range(num_layer):
            if _ == 0:
                irreps_node_in_temp = self.irreps_node_in
            else:
                irreps_node_in_temp = self.irreps_intermediate

            node_update = NodeUpdate(
                irreps_node_in = irreps_node_in_temp,
                irreps_edge_fea = self.irreps_edge_in,
                irreps_out = self.irreps_intermediate,
                irreps_edge_attr = self.irreps_edge_attr,
                fcn_len = self.fcn_len,
                )

            self.node_updates.append(node_update)

        self.sc_pc = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_intermediate,
            irreps_in2=self.irreps_intermediate,
            irreps_out=self.irreps_pc_vec,
        )

    def forward(self,data):
        node_fea = F.relu(self.em(data['x']))
        edge_fea = data['edge_fea']
        edge_src = data['edge_index'][0]  # edge source
        edge_dst = data['edge_index'][1]  # edge destination
        pos = data['pos']
        edge_vec = (pos[edge_dst] - pos[edge_src])
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='gaussian',
            cutoff=False
        ).mul(self.number_of_basis**0.5)
        edge_attr = edge_sh
        for node_update in self.node_updates:
            node_fea = node_update(node_fea,edge_src, edge_dst, edge_fea, edge_attr, edge_length_embedded)
        pc = self.sc_pc(node_fea,node_fea)
        return pc
#############################
##### Eval and training #####
####################################################################################
def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step*(1 - math.exp(-t*rate/step)))
####################################################################################
def train(model, lr, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae,
          run_name, max_iter=101, scheduler=None, device="cpu", weight_ratio=1.0,
          scheduler_patience=None, scheduler_factor=0.5, dataloader_test=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if scheduler is None and scheduler_patience is not None:
        # ReduceLROnPlateau counts "bad epochs" after the best epoch, so we
        # subtract one to make "10 non-improving epochs" behave literally.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=max(scheduler_patience - 1, 0),
        )
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()
    history = []
    s0 = 0

    # Try to load a previous checkpoint.
    try:
        checkpoint_data = torch.load(run_name + '.torch', map_location=device)
        model.load_state_dict(checkpoint_data['state'])
        if checkpoint_data.get('optimizer') is not None:
            optimizer.load_state_dict(checkpoint_data['optimizer'])
        if scheduler is not None and checkpoint_data.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint_data['scheduler'])
    except Exception as e:
        # If no checkpoint exists or loading fails, start from scratch.
        print("No checkpoint found or failed to load checkpoint. Starting from scratch.")
        history = []
        s0 = 0
    else:
        history = checkpoint_data.get('history', [])
        s0 = history[-1]['step'] + 1 if history else 0
        print(f"Resuming training from step {s0}")


    count = 0
    for step in range(max_iter):
        model.train()
        total_loss_cum = 0.0
        for d in tqdm(dataloader_train, total=len(dataloader_train), bar_format=bar_format):
            optimizer.zero_grad()
            d.to(device)
            pred_pc = model(d)
            # flatten the predicted point cloud
            pred_pc = pred_pc.view(-1)
            pred_norm = torch.norm(pred_pc)
            pred_pc = pred_pc / pred_norm
            ref_pc = d.pcs.to(device)[:, 0]

            #if count < 3:
                #print(f"train ref pc sample values: {ref_pc[:5]}")
            count += 1
            # Compute squared norm of ref_pc and the dot product. 
            ref_norm_sq = (ref_pc ** 2).sum()

            

            # Newway to compute loss 
            dot_prod = torch.dot(pred_pc, ref_pc)

            #print(f"train dot prod: {dot_prod.item():.6f}")
            loss = 1.0 - abs(dot_prod)
            # Compute loss (here we use MSE between predicted and target eigenvectors)
            loss.backward()
            optimizer.step()
            total_loss_cum += loss.item()
        avg_loss = total_loss_cum / len(dataloader_train)
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            previous_lr = current_lr
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < previous_lr:
                print(
                    f"Step {s0 + step + 1:4d}: reducing learning rate "
                    f"from {previous_lr:.6f} to {current_lr:.6f}"
                )
        # Checkpoint every so often.
        # Note: The actual training step is s0 + step.
        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            valid_loss = evaluate(model, dataloader_valid, loss_fn, loss_fn_mae, device, weight_ratio)
            test_loss = evaluate(model, dataloader_test, loss_fn, loss_fn_mae, device, weight_ratio) if dataloader_test is not None else float('nan')
            history.append({
                'step': s0 + step,
                'train_loss': avg_loss,
                'valid_loss': valid_loss,
                'test_loss': test_loss,
                'lr': current_lr,
                'time': time.time() - start_time
            })
            print(
                f"Step {s0 + step + 1:4d}: train loss = {avg_loss:.8f}  "
                f"valid loss = {valid_loss:.8f}  test loss = {test_loss:.8f}  "
                f"lr = {current_lr:.6f}  elapsed = {time.time()-start_time:.1f} sec"
            )
            torch.save(
                {
                    'state': model.state_dict(),
                    'history': history,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                },
                run_name + '.torch'
            )
    return history

def evaluate(model, dataloader, loss_fn, loss_fn_mae, device, weight_ratio):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for d in dataloader:
            d.to(device)
            pred_pc = model(d)
            # flatten the predicted point cloud
            pred_pc = pred_pc.view(-1)
            pred_norm = torch.norm(pred_pc)
            pred_pc = pred_pc / pred_norm
            ref_pc = d.pcs.to(device)[:, 0]
            #if count < 3:
                #print(f"eval ref pc sample values: {ref_pc[:5]}")

            # Compute squared norm of ref_pc and the dot product. 
            ref_norm_sq = (ref_pc ** 2).sum()

            # Newway to compute loss 
            dot_prod = torch.dot(pred_pc, ref_pc)

            #print(f"eval dot prod: {dot_prod.item():.6f}")
            loss = 1.0 - abs(dot_prod)
            #print("in eval new loss ",loss) 


            total_loss += loss.item()
            count += 1
            #print(f"re_norm_sq: {ref_norm_sq.item():.6f}, dot_prod: {dot_prod.item():.6f}, loss: {loss.item():.6f}")
    #print(f"count: {count}")
    #print(f"len(dataloader): {len(dataloader)}")
    return total_loss / len(dataloader)
