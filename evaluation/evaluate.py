"""Taken from: https://github.com/pengxingang/Pocket2Mol/blob/main/evaluation/evaluate.py"""
import collections
import argparse
import logging

import os
import torch
import tqdm
from rdkit import Chem
from rdkit.Chem.QED import qed

import docking
import scoring_func


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def read_ligand(ligand_path):
    return next(iter(Chem.SDMolSupplier(ligand_path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_pairs', type=str)
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=1000000)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--exp_name', type=str)
    args = parser.parse_args()

    logger = get_logger('evaluate')
    logger.info(args)

    results = []

    prediction_pairs = torch.load(args.results_pairs)
    prediction_pairs_dict = collections.defaultdict(list)
    gt_pairs = dict()
    for prot, lig in prediction_pairs:
        prediction_pairs_dict[prot].append(lig)
        gt_pairs[prot] = prot.replace('_pocket10.pdb', '.sdf')
    prediction_pairs = dict(prediction_pairs_dict)
    max_index = min(args.end_id, len(gt_pairs))
    indices = range(args.start_id, max_index)
    protein_list = sorted(list(gt_pairs))
    for i in tqdm.tqdm(indices, desc='All'):
        protein = protein_list[i]
        ligands = [gt_pairs[protein], *prediction_pairs[protein]]
        for lig in ligands:
            pair = (protein, lig)
            metrics = {}
            name = args.exp_name
            if not os.path.exists(pair[1]):
                raise FileNotFoundError(pair[1])
            try:
                mol = read_ligand(pair[1])
                vina_task = docking.QVinaDockingTask.from_original_data(pair)
                metrics[name] = {
                    'idx': i,
                    'mol': mol,
                    'vina': vina_task.run_sync(),
                    'qed': qed(mol),
                    'sa': scoring_func.compute_sa_score(mol),
                    'lipinski': scoring_func.obey_lipinski(mol),
                    'logp': scoring_func.get_logp(mol),
                    'protein_path': protein,
                    'ligand_path': lig,
                }
            except Exception as e:
                metrics[name] = {
                    'idx': i,
                    'protein_path': protein,
                    'ligand_path': lig
                }
                logger.warning('Failed %d' % i)
                logger.warning(e)
            results.append(metrics)

    logger.info('Number of results: %d' % len(results))
    result_path = os.path.join(args.out_dir, f'{args.exp_name}_{args.start_id}_{max_index}.pt')
    torch.save(results, result_path)
