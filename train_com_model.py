import argparse
import collections
import os

import numpy as np
import random
import torch
import tqdm

from Bio.PDB import PDBParser
from torch.utils.data import Dataset
from torch_geometric.data import Batch

import com_model
import scoring_model
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=4)
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--n_iter', type=int, default=10000)
    parser.add_argument('--val_freq', type=int, default=100)
    parser.add_argument('--n_samples_per_dir', type=int, default=5)
    # Data
    parser.add_argument('--cross_docked_data_dir', type=str, default='data/crossdocked_pocket10/')
    parser.add_argument('--train_test_split_file', type=str, default='data/split_by_name.pt')
    # Output
    parser.add_argument('--output_checkpoint_name', type=str, default='com_model_retrained')
    return parser.parse_args()


class ProteinGraphDataset(Dataset):
    def __init__(self, protein_list):
        super().__init__()
        self.protein_list = protein_list

    def __getitem__(self, idx):
        return self.protein_list[idx]

    def __len__(self):
        return len(self.protein_list)


def get_unique_pocket_dirs(protein_ligand_pairs):
    all_pocket_dirs = set()
    for prot_path, lig_path in protein_ligand_pairs:
        prot_dir = os.path.split(prot_path)[0]
        all_pocket_dirs.add(prot_dir)
    return list(all_pocket_dirs)


def get_train_val_pairs(pocket_dirs, protein_ligand_pairs):
    train_pocket_dirs = random.sample(pocket_dirs, k=int(0.8 * len(pocket_dirs)))
    train_pairs = []
    val_pairs = []
    for prot_path, lig_path in protein_ligand_pairs:
        prot_dir = os.path.split(prot_path)[0]  # split train/val by pocket dir to avoid data leak
        if prot_dir in train_pocket_dirs:
            train_pairs.append((prot_path, lig_path))
        else:
            val_pairs.append((prot_path, lig_path))
    return train_pairs, val_pairs


def get_pocket_dir_to_pair_mapping(protein_ligand_pairs):
    pocket_dir_to_pair_mapping = collections.defaultdict(list)
    for prot_path, lig_path in protein_ligand_pairs:
        prot_dir = os.path.split(prot_path)[0]
        pocket_dir_to_pair_mapping[prot_dir].append((prot_path, lig_path))
    return dict(pocket_dir_to_pair_mapping)


def get_uniform_pairs_sample_per_pocket_dir(pocket_dir_to_pair_mapping, n_samples_per_prot_dir):
    pairs_sample = []
    for prot_dir, sample in pocket_dir_to_pair_mapping.items():
        n_samples = min(n_samples_per_prot_dir, len(sample))
        pairs_sample.extend(random.sample(sample, k=n_samples))
    return pairs_sample


def get_protein_and_ligand_datasets(protein_ligand_pairs, parser, crossdocked_data_dir):
    protein_graphs = []
    coms = []
    for prot_path, lig_path in tqdm.tqdm(protein_ligand_pairs):
        try:
            protein_graph = scoring_model.get_protein_graph(
                utils.load_protein(parser, os.path.join(crossdocked_data_dir, prot_path)),
                utils.RESIDUE_ENCODER, utils.RESIDUE_NAME_INDEX)
            protein_graphs.append(protein_graph)
        except KeyError:  # PDB file misunderstood for PDBParser
            continue
        ligand = utils.load_ligand_sdf(os.path.join(crossdocked_data_dir, lig_path))
        com = np.mean(ligand.GetConformer().GetPositions(), axis=0)
        coms.append(torch.from_numpy(com))
    return ProteinGraphDataset(protein_graphs), torch.stack(coms, dim=0).float()


def get_batch(protein_dataset, coms, batch_size):
    batch_indices = random.sample(list(range(len(protein_dataset))), k=batch_size)
    protein_batch = Batch.from_data_list([protein_dataset[i] for i in batch_indices])
    com_batch = coms[batch_indices]
    return protein_batch, com_batch


if __name__ == "__main__":
    args = parse_args()
    train_test_split = torch.load(args.train_test_split_file)
    all_pairs = train_test_split['train']
    all_pocket_dirs = get_unique_pocket_dirs(all_pairs)
    train_pairs, val_pairs = get_train_val_pairs(all_pocket_dirs, all_pairs)
    biopython_parser = PDBParser()
    train_pairs_sample = get_uniform_pairs_sample_per_pocket_dir(get_pocket_dir_to_pair_mapping(train_pairs),
                                                                 n_samples_per_prot_dir=args.n_samples_per_dir)
    val_pairs_sample = random.sample(val_pairs, k=args.val_batch_size)
    print("Creating datasets...", end='')
    train_protein_dataset, train_coms = get_protein_and_ligand_datasets(train_pairs_sample,
                                                                        biopython_parser,
                                                                        args.cross_docked_data_dir)
    val_protein_dataset, val_coms = get_protein_and_ligand_datasets(val_pairs_sample,
                                                                    biopython_parser,
                                                                    args.cross_docked_data_dir)
    print("done.")
    val_protein_batch, val_coms_batch = get_batch(val_protein_dataset, val_coms, batch_size=args.val_batch_size)
    model = com_model.CoMPredictor(in_dim=len(utils.RESIDUE_NAME_INDEX),
                                   hidden_dim=args.hidden_dim,
                                   out_dim=1,
                                   n_layers=args.n_layers)
    opt = torch.optim.Adam(model.parameters())
    losses = []
    for batch_idx in tqdm.tqdm(range(args.n_iter)):
        train_protein_batch, coms_train_batch = get_batch(train_protein_dataset, train_coms, batch_size=args.batch_size)
        preds = model(train_protein_batch)
        loss = ((preds - coms_train_batch) ** 2).sum(dim=1).sqrt().mean()
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (batch_idx % args.val_freq) == 0:
            with torch.no_grad():
                preds_test = model(val_protein_batch)
                test_loss = ((preds_test - val_coms_batch) ** 2).sum(dim=1).sqrt().mean()
            print(f'Tr loss: {np.mean(losses[-args.val_freq:]):.3f} Val loss: {test_loss.item():.3f}')

    model_output_path = f'checkpoints/{args.output_checkpoint_name}_state_dict.pt'
    print(f'Finished training. Saving model to {model_output_path}')
    torch.save(model.state_dict(), model_output_path)
