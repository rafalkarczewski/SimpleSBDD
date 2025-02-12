import os
import torch
import numpy as np
from rdkit import Chem

from moflow.data.transform_zinc250k import zinc250_atomic_num_list
from moflow.data.smile_to_graph import construct_discrete_edge_matrix
from moflow.mflow.models.hyperparams import Hyperparameters
from moflow.mflow.models.utils import check_validity
from moflow.mflow.utils.model_utils import load_model


def get_adj(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    adj = construct_discrete_edge_matrix(mol, out_size=38)
    return np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0).astype(np.float32)


def generate_mols(model, temp=0.7, z_mu=None, batch_size=20, true_adj=None, device=-1):  #  gpu=-1):
    # xp = np
    if isinstance(device, torch.device):
        # xp = chainer.backends.cuda.cupy
        pass
    elif isinstance(device, int):
        if device >= 0:
            # device = args.gpu
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', int(device))
        else:
            device = torch.device('cpu')
    else:
        raise ValueError("only 'torch.device' or 'int' are valid for 'device', but '%s' is "'given' % str(device))

    z_dim = model.b_size + model.a_size  # 324 + 45 = 369   9*9*4 + 9 * 5
    mu = np.zeros(z_dim)  # (369,) default , dtype=np.float64
    sigma_diag = np.ones(z_dim)  # (369,)

    if model.hyper_params.learn_dist:
        if len(model.ln_var) == 1:
            sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag
        elif len(model.ln_var) == 2:
            sigma_diag[:model.b_size] = np.sqrt(np.exp(model.ln_var[0].item())) * sigma_diag[:model.b_size]
            sigma_diag[model.b_size+1:] = np.sqrt(np.exp(model.ln_var[1].item())) * sigma_diag[model.b_size+1:]

        # sigma_diag = xp.exp(xp.hstack((model.ln_var_x.data, model.ln_var_adj.data)))

    sigma = temp * sigma_diag

    with torch.no_grad():
        if z_mu is not None:
            mu = z_mu
            sigma = 0.01 * np.eye(z_dim)
        # mu: (369,), sigma: (369,), batch_size: 100, z_dim: 369
        z = np.random.normal(mu, sigma, (batch_size, z_dim))  # .astype(np.float32)
        z = torch.from_numpy(z).float().to(device)
        adj, x = model.reverse(z, true_adj=true_adj)

    return adj, x  # (bs, n_bond_types, max_num_atoms, max_num_atoms), (bs, max_num_atoms, num_atom_types)


def get_moflow_resampler(model_dir, snapshot_path, hyperparams_path, temperature):
    hyperparams_path = os.path.join(model_dir, hyperparams_path)
    snapshot_path = os.path.join(model_dir, snapshot_path)
    model_params = Hyperparameters(path=hyperparams_path)
    model = load_model(snapshot_path, model_params)
    model.eval()
    atomic_num_list = zinc250_atomic_num_list

    def resample(smiles_list):
        adj_array = []
        for smile in smiles_list:
            adj_array.append(get_adj(smile))
        adj_array = torch.from_numpy(np.array(adj_array))
        batch_size = adj_array.shape[0]
        adj, x = generate_mols(model, batch_size=batch_size, true_adj=adj_array, temp=temperature)
        val_res = check_validity(adj, x, atomic_num_list, correct_validity=True, return_unique=False)
        return val_res['valid_smiles']
    return resample


def get_moflow_prop_optimizer(model_dir, snapshot_path, hyperparams_path, temperature, n_trials, property_fn):
    hyperparams_path = os.path.join(model_dir, hyperparams_path)
    snapshot_path = os.path.join(model_dir, snapshot_path)
    model_params = Hyperparameters(path=hyperparams_path)
    model = load_model(snapshot_path, model_params)
    model.eval()
    atomic_num_list = zinc250_atomic_num_list

    def optimize(smiles_string):
        true_adj = torch.from_numpy(get_adj(smiles_string)).unsqueeze(0).repeat((n_trials, 1, 1, 1))
        adj, x = generate_mols(model, batch_size=n_trials, true_adj=true_adj, temp=temperature)
        val_res = check_validity(adj, x, atomic_num_list, correct_validity=False,
                                 return_unique=True)
        return max(val_res['valid_smiles'], key=lambda s: property_fn(Chem.MolFromSmiles(s)))

    def resample(smiles_list):
        return [optimize(smiles_string) for smiles_string in smiles_list]

    return resample
