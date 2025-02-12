import argparse
import os
import random
import time

import numpy as np
import torch
import tqdm
import scoring_model
import com_model
import utils

from evaluation_utils import compute_sa_score

from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem.QED import qed
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from moflow_sampling import get_moflow_resampler, get_moflow_prop_optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--smiles_sample_size', type=int, default=2 ** 14)
    parser.add_argument('--n_samples_per_pocket', type=int, default=100)
    parser.add_argument('--min_percentile', type=int, default=5)
    parser.add_argument('--max_percentile', type=int, default=10)
    parser.add_argument('--model_type', type=str, default='DR', choices=['DR', 'ND', 'PO'])
    parser.add_argument('--n_trials_po', type=int, default=50)
    # MoFlow Params
    parser.add_argument('--moflow_model_dir', type=str, default='moflow/mflow/results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask')
    parser.add_argument('--moflow_temperature', type=float, default=0.85)
    parser.add_argument('--moflow_hyper_params_path', type=str, default='moflow-params.json')
    parser.add_argument('--moflow_snapshot', type=str, default='model_snapshot_epoch_200')
    # Data
    parser.add_argument('--train_test_pairs_file_path', type=str, default='data/split_by_name.pt')
    parser.add_argument('--smiles_file_path', type=str, default='data/ZINC.txt')
    parser.add_argument('--scoring_model_ckpt_path', type=str, default='checkpoints/scoring_model_state_dict.pt')
    parser.add_argument('--com_model_ckpt_path', type=str, default='checkpoints/com_model_state_dict.pt')
    # Output
    parser.add_argument('--cross_docked_data_dir', type=str, default='data/crossdocked_pocket10/')
    parser.add_argument('--experiment_name', type=str, default='fastsbdd_drug_repurposing')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scorer = scoring_model.ScoringModel(
        protein_kwargs={'in_dim': len(utils.RESIDUE_NAME_INDEX),
                        'hidden_dim': 16,
                        'out_dim': 16,
                        'n_layers': 3},
        mol_dim=4,
        n_mixing_layers=4,
        hidden_dim=50
    )
    scorer.load_state_dict(torch.load(args.scoring_model_ckpt_path))
    com_predictor = com_model.CoMPredictor(in_dim=20, hidden_dim=16, out_dim=1, n_layers=4)
    smiles = []
    with open(args.smiles_file_path, 'r') as f:
        for line in f:
            smiles.append(line[:-1])
    candidates = random.sample(smiles, k=args.smiles_sample_size)
    t1 = time.time()
    candidates_features = torch.stack([
        scoring_model.get_ligand_features(Chem.MolFromSmiles(smile))
        for smile in candidates], dim=0).float()
    print(f'Featurizing {args.smiles_sample_size} mols took: {(time.time() - t1):.2} seconds')
    test_pairs = torch.load(args.train_test_pairs_file_path)['test']
    biopython_parser = PDBParser()
    experiment_dir = os.path.join('results', f'{args.experiment_name}_experiment')
    os.makedirs(experiment_dir)

    eval_pairs = []
    gen_times = []
    conf_times = []

    if args.model_type == 'DR':
        resampler = lambda x: x
    elif args.model_type == 'ND':
        resampler = get_moflow_resampler(args.moflow_model_dir,
                                         args.moflow_snapshot,
                                         args.moflow_hyper_params_path,
                                         args.moflow_temperature)
    else:
        def property_fn(mol):
            return 5 * qed(mol) + compute_sa_score(mol)
        resampler = get_moflow_prop_optimizer(args.moflow_model_dir,
                                              args.moflow_snapshot,
                                              args.moflow_hyper_params_path,
                                              args.moflow_temperature,
                                              n_trials=args.n_trials_po,
                                              property_fn=property_fn)
    eval_pairs = []
    for idx, (test_prot_path, test_lig_path) in tqdm.tqdm(enumerate(test_pairs)):
        t1 = time.time()
        pocket_ligands_dir = f'{experiment_dir}/{idx}'
        test_prot_full_path = os.path.join(args.cross_docked_data_dir, test_prot_path)
        os.makedirs(pocket_ligands_dir, exist_ok=True)
        protein_pocket = utils.load_protein(biopython_parser, test_prot_full_path)
        protein_pocket_graph = scoring_model.get_protein_graph(
            protein_pocket, utils.RESIDUE_ENCODER, utils.RESIDUE_NAME_INDEX)
        test_scores = scorer.score_single_pocket(protein_pocket_graph, candidates_features)
        predicted_com = com_predictor(protein_pocket_graph).detach().numpy()[0]
        vmin, vmax = np.percentile(test_scores, q=[args.min_percentile, args.max_percentile])
        good_candidates = np.where(np.logical_and(test_scores > vmin, test_scores < vmax))[0]
        selected_smiles = random.sample([candidates[idx] for idx in good_candidates.tolist()],
                                        k=args.n_samples_per_pocket)
        selected_smiles = resampler(selected_smiles)
        for sample_idx, smiles_string in enumerate(selected_smiles):
            sample_mol = Chem.MolFromSmiles(smiles_string)
            try:
                utils.generate_conformer(sample_mol)
            except ValueError:
                continue
            utils.translate_mol(sample_mol, predicted_com)
            lig_save_path = f'{experiment_dir}/{idx}/mol{sample_idx + 1}.sdf'
            utils.save_ligand(sample_mol, lig_save_path)
            eval_pairs.append((test_prot_path, lig_save_path))
    torch.save(eval_pairs, f'results/{args.experiment_name}_experiment_eval_input.pt')
