import collections
import itertools
import os
import sys

import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def get_binding_affinity(res):
    if 'vina' in res:
        return res['vina'][0]['affinity']
    else:
        return None


def parse_gt_results(results, aggregate):
    gt_metrics = {'ba': [], 'high_affinity': [], 'qed': [], 'sa': [], 'lipinski': [], 'logp': [],
                  'diversity': [], 'mol_size': [], 'pcntg_vina_computed': []}

    for protein_filename, res in results.items():
        gt_metrics['ba'].append(get_binding_affinity(res['GT']))
        gt_metrics['qed'].append(res['GT']['qed'])
        gt_metrics['sa'].append(res['GT']['sa'])
        gt_metrics['lipinski'].append(res['GT']['lipinski'])
        gt_metrics['logp'].append(res['GT']['logp'])
        gt_metrics['mol_size'].append(res['GT']['mol'].GetNumAtoms())
        gt_metrics['pcntg_vina_computed'].append(1.)
        gt_metrics['high_affinity'].append(1.)
        gt_metrics['diversity'].append(0.)
    if aggregate:
        return {
            key: (np.mean(val), np.std(val))
            for key, val in gt_metrics.items()
        }
    else:
        return gt_metrics


def parse_pred_results(results, aggregate, compute_diversity):
    pred_metrics = {'ba': [], 'high_affinity': [], 'qed': [], 'sa': [], 'lipinski': [], 'logp': [],
                    'diversity': [], 'mol_size': [], 'pcntg_vina_computed': []}
    for protein_filename, res in results.items():
        all_bas = [(idx, get_binding_affinity(el)) for idx, el in enumerate(res['pred'])]
        bas = [(idx, el) for idx, el in all_bas if el is not None]
        gt_ba = get_binding_affinity(res['GT'])
        bas_lt_gt = []
        mols = []
        for idx, ba in bas:
            bas_lt_gt.append(ba <= gt_ba)
            pred_metrics['ba'].append(ba)
            pred_metrics['qed'].append(res['pred'][idx]['qed'])
            pred_metrics['sa'].append(res['pred'][idx]['sa'])
            pred_metrics['logp'].append(res['pred'][idx]['logp'])
            pred_metrics['lipinski'].append(res['pred'][idx]['lipinski'])
            pred_metrics['mol_size'].append(res['pred'][idx]['mol'].GetNumAtoms())
            pred_metrics['pcntg_vina_computed'].append(len(bas) / len(all_bas))
            mols.append(res['pred'][idx]['mol'])
        if compute_diversity:
            pred_metrics['diversity'].append(calculate_diversity(mols))
        else:
            pred_metrics['diversity'].append(0)
        if bas_lt_gt:
            pred_metrics['high_affinity'].append(np.mean(bas_lt_gt))
    if aggregate:
        return {
            key: (np.mean(val), np.std(val))
            for key, val in pred_metrics.items()
        }
    else:
        return pred_metrics


def results_from_files(file_list, exp_name, gt=False, aggregate=True, parse=True, compute_diversity=True):
    _results = list(itertools.chain.from_iterable(file_list))
    results = collections.defaultdict(dict)

    for res in _results:
        res_dict = res[exp_name]
        try:
            protein_filename = res_dict['protein_path']
            ligand_filename = res_dict['ligand_path']
        except KeyError:
            continue
        new_result = results[protein_filename]
        if protein_filename.startswith(ligand_filename[:-4]): # ground truth
            new_result['GT'] = res_dict
        else:
            preds = new_result.get('pred', [])
            preds.append(res_dict)
            new_result['pred'] = preds
    results = dict(results)
    if not parse:
        return results
    if gt:
        return parse_gt_results(results, aggregate)
    else:
        return parse_pred_results(results, aggregate, compute_diversity)


def similarity(mol_a, mol_b):
    fp1 = Chem.RDKFingerprint(mol_a)
    fp2 = Chem.RDKFingerprint(mol_b)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def calculate_diversity(pocket_mols):
    if len(pocket_mols) < 2:
        return 0.0
    div = 0
    total = 0
    for i in range(len(pocket_mols)):
        for j in range(i + 1, len(pocket_mols)):
            div += 1 - similarity(pocket_mols[i], pocket_mols[j])
            total += 1
    return div / total


def compute_sa_score(rdmol):
    rdmol = Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))
    sa = sascorer.calculateScore(rdmol)
    sa_norm = round((10-sa)/9,2)
    return sa_norm
