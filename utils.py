import warnings

import numpy as np

from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

ps = AllChem.ETKDGv2()

RESIDUE_NAME_INDEX ={
    'GLY': 0, 'ASP': 1, 'MET': 2, 'SER': 3, 'CYS': 4, 'PHE': 5, 'VAL': 6,
    'GLU': 7, 'THR': 8, 'PRO': 9, 'HIS': 10, 'ARG': 11, 'ALA': 12, 'ILE': 13,
    'TYR': 14, 'LEU': 15, 'ASN': 16, 'LYS': 17, 'TRP': 18, 'GLN': 19}

RESIDUE_ENCODER = np.eye(len(RESIDUE_NAME_INDEX))


def load_ligand_sdf(ligand_filepath):
    return Chem.SDMolSupplier(ligand_filepath)[0]


def load_protein(parser, protein_filepath):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        return parser.get_structure('random_id', protein_filepath)[0]


def save_ligand(ligand, ligand_path):
    sdf_writer = Chem.SDWriter(ligand_path)
    sdf_writer.write(ligand)
    sdf_writer.close()


def translate_mol(mol, target_com):
    "Changing mol in-place"
    current_com = np.mean(mol.GetConformer().GetPositions(), axis=0)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        current_pos = conf.GetAtomPosition(i)
        translation_vector = target_com - current_com
        new_pos = current_pos + Point3D(*translation_vector.tolist())
        conf.SetAtomPosition(i, new_pos)


def generate_conformer(mol):
    "Generates a conformer and modifies mol in-place"
    if AllChem.EmbedMolecule(mol, ps) != 0:
        raise ValueError("Unable to generate a conformer for mol")
    AllChem.UFFOptimizeMolecule(mol, confId=0)
