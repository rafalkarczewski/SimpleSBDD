import sys

import numpy as np
import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_geometric.nn import aggr
# sys.path.append('../../hsbdd/Modular-Flows-Neurips-2022/')
import egnn_clean as eg

mean_aggr = aggr.MeanAggregation()


class ScoringModel(nn.Module):
    def __init__(self, protein_kwargs, mol_dim, n_mixing_layers, hidden_dim, mol_mean=None, mol_std=None):
        super().__init__()
        mol_mean = mol_mean if mol_mean is not None else torch.zeros(size=(1, mol_dim))
        mol_std = mol_std if mol_std is not None else torch.zeros(size=(1, mol_dim))
        self.mol_mean = nn.Parameter(mol_mean, requires_grad=False)
        self.mol_std = nn.Parameter(mol_std, requires_grad=False)
        self.protein_layer = eg.EGNN(in_node_nf=protein_kwargs['in_dim'],
                                     hidden_nf=protein_kwargs['hidden_dim'],
                                     out_node_nf=protein_kwargs['out_dim'],
                                     n_layers=protein_kwargs['n_layers'])
        combined_dim = protein_kwargs['out_dim'] + mol_dim
        dims = [(combined_dim, hidden_dim)] + [
            (hidden_dim, hidden_dim)
            for _ in range(n_mixing_layers - 1)
        ]
        self.mlp_layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in dims])
        self.final_layer = nn.Linear(hidden_dim, 1)

    def encode_protein(self, protein_data_inp):
        protein_data = protein_data_inp.clone()
        edges = [protein_data.edge_index[0].long(), protein_data.edge_index[1].long()]
        h, pos = self.protein_layer(protein_data.x.float(), protein_data.pos.float(), edges, None)
        return mean_aggr(h, protein_data.batch)

    def mlp_forward(self, protein_repr, mol_data):
        mol_repr = (mol_data - self.mol_mean) / self.mol_std
        combined_repr = torch.cat([protein_repr, mol_repr], dim=1)
        for mixing_layer in self.mlp_layers:
            combined_repr = torch.relu(mixing_layer(combined_repr))
        return self.final_layer(combined_repr)

    def forward(self, protein_data_inp, mol_data):
        prot_repr = self.encode_protein(protein_data_inp)
        return self.mlp_forward(prot_repr, mol_data)

    def score_single_pocket(self, single_pocket, mol_data):
        with torch.no_grad():
            prot_repr = self.encode_protein(single_pocket).repeat((mol_data.shape[0], 1))
            return self.mlp_forward(prot_repr, mol_data)[:, 0].detach().numpy()


def get_ligand_features(mol):
    n_atoms = mol.GetNumAtoms()
    n_rings = mol.GetRingInfo().NumRings()
    n_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    diameter = np.max(Chem.GetDistanceMatrix(mol))
    return torch.tensor([n_atoms, n_rings, n_rot_bonds, diameter])


def get_protein_graph(protein, res_encoder, res_index_map, distance_cutoff=15, max_num_neighbours=24):
    res_encodings = []
    pos_encodings = []
    for residue in protein.get_residues():
        res_encodings.append(res_encoder[res_index_map[residue.get_resname()]])
        pos_encodings.append(residue.center_of_mass())
    res_one_hot = torch.from_numpy(np.array(res_encodings))
    pos = torch.from_numpy(np.array(pos_encodings))
    edge_index = radius_graph(pos, distance_cutoff, max_num_neighbors=max_num_neighbours)
    return Data(x=res_one_hot, edge_index=edge_index, pos=pos)
