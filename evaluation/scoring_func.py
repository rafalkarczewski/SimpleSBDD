"""Taken from https://github.com/pengxingang/Pocket2Mol/blob/main/evaluation/scoring_func.py"""
import copy
import os
import sys

import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def obey_lipinski(mol):
    mol = copy.deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp:=Crippen.MolLogP(mol)>=-2) & (logp<=5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def get_logp(mol):
    return Crippen.MolLogP(mol)


def compute_sa_score(mol):
    return sascorer.calculateScore(mol)
