import argparse
import os

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import tqdm

from Bio.PDB import PDBParser
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.data import Batch

import scoring_model
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--protein_encoder_hidden_dim', type=int, default=16)
    parser.add_argument('--protein_encoder_out_dim', type=int, default=16)
    parser.add_argument('--protein_encoder_n_layers', type=int, default=3)
    parser.add_argument('--mlp_n_layers', type=int, default=4)
    parser.add_argument('--mlp_hidden_dim', type=int, default=50)
    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_iter', type=int, default=10000)
    parser.add_argument('--val_freq', type=int, default=100)
    # Data
    parser.add_argument('--cross_docked_data_dir', type=str, default='data/crossdocked_pocket10/')
    parser.add_argument('--bining_affinity_data_path', type=str, default='data/binding_affinity.csv')
    # Output
    parser.add_argument('--output_checkpoint_name', type=str, default='scoring_model_retrained')
    return parser.parse_args()


class ProteinGraphDataset(Dataset):
    def __init__(self, protein_list):
        super().__init__()
        self.protein_list = protein_list

    def __getitem__(self, idx):
        return self.protein_list[idx]

    def __len__(self):
        return len(self.protein_list)


def get_triples_dataset(dataset_df, protein_path_mapping):
    dataset_triples = []
    all_mol_features = []
    for idx, row in dataset_df.iterrows():
        vina_score = row['vina_score']
        protein_idx = protein_path_mapping[row['protein_path']]
        mol_features = scoring_model.get_ligand_features(Chem.MolFromSmiles(row['ligand_smiles']))
        mol_feature_idx = len(all_mol_features)
        all_mol_features.append(mol_features)
        dataset_triples.append((protein_idx, mol_feature_idx, vina_score))
    all_mol_features = torch.stack(all_mol_features).float()
    return dataset_triples, all_mol_features


def get_batch(triples_dataset, protein_dataset, mol_features, batch_size):
    batch = random.sample(triples_dataset, k=batch_size)
    protein_indices, mol_indices, vina_scores = zip(*batch)
    protein_batch = Batch.from_data_list([protein_dataset[i] for i in protein_indices])
    mol_batch = mol_features[list(mol_indices)]
    y_batch = torch.tensor(vina_scores)
    return protein_batch, mol_batch, y_batch


if __name__ == "__main__":
    args = parse_args()
    biopython_parser = PDBParser()

    dataset = pd.read_csv(args.bining_affinity_data_path)
    train_dataset = dataset[dataset.split_name == 'TRAINING']
    val_dataset = dataset[dataset.split_name == 'VALIDATION']
    train_protein_path_list = list(set(train_dataset['protein_path'].tolist()))
    train_protein_path_mapping = {prot_path: idx for idx, prot_path in enumerate(train_protein_path_list)}
    val_protein_path_list = list(set(val_dataset['protein_path'].tolist()))
    val_protein_path_mapping = {prot_path: idx for idx, prot_path in enumerate(val_protein_path_list)}

    print("Creating graph representations of proteins...", end='')
    train_protein_dataset = ProteinGraphDataset(
        [
            scoring_model.get_protein_graph(utils.load_protein(
                biopython_parser,
                os.path.join(args.cross_docked_data_dir, local_prot_path)),
                utils.RESIDUE_ENCODER,
                utils.RESIDUE_NAME_INDEX)
            for local_prot_path in train_protein_path_list
        ]
    )

    val_protein_dataset = ProteinGraphDataset(
        [
            scoring_model.get_protein_graph(utils.load_protein(
                biopython_parser,
                os.path.join(args.cross_docked_data_dir, local_prot_path)),
                utils.RESIDUE_ENCODER,
                utils.RESIDUE_NAME_INDEX)
            for local_prot_path in val_protein_path_list
        ]
    )
    print('done')

    print('Creating training and validation datasets...', end='')
    train_dataset_triples, train_all_mol_features = get_triples_dataset(train_dataset, train_protein_path_mapping)
    val_dataset_triples, val_all_mol_features = get_triples_dataset(val_dataset, val_protein_path_mapping)
    print('done.')

    mean_mol_features = train_all_mol_features.mean(dim=0, keepdim=True)
    std_mol_features = train_all_mol_features.std(dim=0, keepdim=True)

    model = scoring_model.ScoringModel(
        protein_kwargs={'in_dim': len(utils.RESIDUE_NAME_INDEX),
                        'hidden_dim': args.protein_encoder_hidden_dim,
                        'out_dim': args.protein_encoder_out_dim,
                        'n_layers': args.protein_encoder_n_layers},
        mol_dim=mean_mol_features.shape[1],
        mol_mean=mean_mol_features,
        mol_std=std_mol_features,
        n_mixing_layers=args.mlp_n_layers,
        hidden_dim=args.mlp_hidden_dim
    )

    protein_test_batch, mol_test_batch, y_test_batch = get_batch(
        val_dataset_triples, val_protein_dataset, val_all_mol_features, batch_size=len(val_dataset_triples))

    loss_fn = nn.MSELoss(reduction='mean')
    opt = torch.optim.Adam(model.parameters())
    losses = []
    print('Starting training')
    for batch_idx in tqdm.tqdm(range(args.n_iter)):
        protein_batch, mol_batch, y_batch = get_batch(
            train_dataset_triples, train_protein_dataset, train_all_mol_features,
            batch_size=args.batch_size)
        preds = model(protein_batch, mol_batch)
        loss = loss_fn(preds[:, 0], y_batch)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (batch_idx % args.val_freq) == 0:
            with torch.no_grad():
                preds_test = model(protein_test_batch, mol_test_batch)
                test_loss = loss_fn(preds_test[:, 0], y_test_batch)
            print(f'Tr loss: {np.mean(losses[-args.val_freq:]):.3f} Val loss: {test_loss.item():.3f}')

    model_output_path = f'checkpoints/{args.output_checkpoint_name}_state_dict.pt'
    print(f'Finished training. Saving model to {model_output_path}')
    torch.save(model.state_dict(), model_output_path)
