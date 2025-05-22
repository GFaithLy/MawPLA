from torch.utils.data import Dataset
#from ligand import get_morgan_fingerprint_dict,ligand_init
#from use import esm
import pickle
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import torch
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree, add_self_loops, subgraph, to_undirected, remove_self_loops, coalesce
from rdkit import Chem
from pathlib import Path
from collections import namedtuple
from rdkit import Chem
from rdkit.Chem import AllChem

device = "cuda:1" if torch.cuda.is_available() else "cpu"

def validate_smiles(smiles_list):
    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
    return valid_smiles
def get_tree_info_from_ligand_dict(ligand_info_dict):
    """
    :param ligand_info_dict:Dictionary of ligand information returned by the ligand_init function
    :return: Dictionary with information about the connection tree, containing 'tree_edge_index', 'atom2clique_index', 'num_cliques' and 'x_clique' key-value pairs
    """
    return {
        'tree_edge_index': ligand_info_dict['tree_edge_index'],
        'atom2clique_index': ligand_info_dict['atom2clique_index'],
        'num_cliques': ligand_info_dict['num_cliques'],
        'x_clique': ligand_info_dict['x_clique']
    }


def process_smiles(smi,ligand_info_dict):
    """
    Functions for processing SMILES strings, including validation, obtaining ligand information, extracting atomic-level features and molecular-level features, respectively, and outputting them in a suitable format.

    :param smiles_data: SMILES
    :return: Tuple containing atomic-level features and molecular-level features, atom_feature_data (atomic-level features) and molecular_features (molecular-level features), respectively
    """
    
    # 验证字符串
    valid_smiles = validate_smiles([smi])
    if not valid_smiles:
        raise ValueError("Invalid SMILES string provided.")

    ligand_info_dict = ligand_info_dict[smi]
    
    atom_degree = ligand_info_dict["atom_feature"][:, 0]
    atom_total_valency = ligand_info_dict["atom_feature"][:, 1]
    atom_hybridization = ligand_info_dict["atom_feature"][:, 2:8]
    atom_radical_electrons = ligand_info_dict["atom_feature"][:, 8]
    atom_formal_charge = ligand_info_dict["atom_feature"][:, 9]
    atom_aromatic = ligand_info_dict["atom_feature"][:, 10]
    atom_in_ring = ligand_info_dict["atom_feature"][:, 11]
    atom_classes = ligand_info_dict["atom_feature"][:, 12]
    atom_donor_acceptor_hydrophobe = ligand_info_dict["atom_feature"][:, 13:17]
    try:
        chirality_encoding = ligand_info_dict["atom_feature"][:, 17:20]
    except IndexError:
        chirality_encoding = torch.zeros((atom_degree.shape[0], 3))
    bond_feature = ligand_info_dict["bond_feature"]
    
    tree_info = get_tree_info_from_ligand_dict(ligand_info_dict)
    x_clique = tree_info["x_clique"]
    bond_feature_size = bond_feature.shape[0]
    x_clique_size = x_clique.shape[0]

    if bond_feature_size!= x_clique_size:
        if bond_feature_size > x_clique_size:
            pad_size = bond_feature_size - x_clique_size
            x_clique = F.pad(x_clique, (0, pad_size))
        else:
            x_clique = x_clique[:bond_feature_size]
    # cat
    fused_atom_features = torch.cat((
        atom_degree.unsqueeze(1),
        atom_total_valency.unsqueeze(1),
        atom_hybridization,
        atom_radical_electrons.unsqueeze(1),
        atom_formal_charge.unsqueeze(1),
        atom_aromatic.unsqueeze(1),
        atom_in_ring.unsqueeze(1),
        atom_classes.unsqueeze(1),
        atom_donor_acceptor_hydrophobe,
        chirality_encoding,
        bond_feature,
        x_clique.unsqueeze(1) if len(x_clique.shape) == 1 else x_clique
    ), dim=1)

    fused_atom_features = fused_atom_features.view(-1, 1) if len(fused_atom_features.shape) == 1 else fused_atom_features
    fused_atom_features = fused_atom_features.transpose(0, 1)

    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)
        bit_string_list = [int(bit) for bit in morgan_fingerprint.ToBitString()]
        morgan_fingerprint_tensor = torch.tensor(bit_string_list, dtype=torch.float32).view(1, -1)
    else:
        raise ValueError("Failed to obtain molecule object from the given SMILES string for Morgan fingerprint generation.")

    if fused_atom_features.shape[1] > 1024:
        fused_atom_features = fused_atom_features[:, :1024]
    elif fused_atom_features.shape[1] < 1024:
        if len(fused_atom_features.shape) == 1:
            fused_atom_features = fused_atom_features.unsqueeze(0)
        pad_size = 1024 - fused_atom_features.shape[1]
        fused_atom_features = torch.nn.functional.pad(fused_atom_features, (0, pad_size))

    if morgan_fingerprint_tensor.shape[1] > 1024:
        morgan_fingerprint_tensor = morgan_fingerprint_tensor[:, :1024]
    elif morgan_fingerprint_tensor.shape[1] < 1024:
        pad_size = 1024 - morgan_fingerprint_tensor.shape[1]
        morgan_fingerprint_tensor = torch.nn.functional.pad(morgan_fingerprint_tensor, (0, pad_size))

    # weighted fusion
    alpha = 0.3
    beta = 0.7
    fused_features = alpha * fused_atom_features + beta * morgan_fingerprint_tensor

    if fused_features.shape[0] > 256:
        fused_features = fused_features[:256, :]
    elif fused_features.shape[0] < 256:
        pad_size_0 = 256 - fused_features.shape[0]
        fused_features = torch.nn.functional.pad(fused_features, (0, 0, 0, pad_size_0))

    if fused_features.shape[1] > 256:
        fused_features = fused_features[:, :256]
    elif fused_features.shape[1] < 256:
        pad_size_1 = 256 - fused_features.shape[1]
        fused_features = torch.nn.functional.pad(fused_features, (0, pad_size_1, 0, 0))

    return fused_features 

import pickle
def process_protein(seq,protein_info_dict):
    """
    Function to process protein sequence related information, fetching all required protein data from the saved protein_cache.pkl file.
    :param seq: protein sequence
    :return: data containing sequence attributes and graph attributes (combined_data)
    """
    if seq not in protein_info_dict:
        raise KeyError(f"Sequence {seq} not found in the loaded protein_result.pt file.")
    protein_info_dict = protein_info_dict[seq]

    token_representation = protein_info_dict["token_representation"]#token_repr.half()
    seq_feat = protein_info_dict["seq_feat"].numpy()
    edge_index = protein_info_dict["edge_index"]
    edge_weight = protein_info_dict["edge_weight"]
    num_pos = protein_info_dict["num_pos"]

    # Constructed Diagram Related Information
    num_nodes = len(seq)
    graph_info = {
        "num_nodes": num_nodes,
        
        "token_representation": token_representation.half(),
        "num_pos": num_pos,
        "edge_index": edge_index,
        "edge_weight": edge_weight
    }
    # Constructing data containing sequence attributes and graph attributes
    combined_data = {
        "sequence": {
            "seq": seq,
            "seq_feat": torch.from_numpy(seq_feat)
        },
        "graph": graph_info
    }

    return combined_data
data_cache={}  
device = "cuda:1" if torch.cuda.is_available() else "cpu"
class MyDataset(Dataset):
    def __init__(self, type, data_path, max_seq_len, max_smi_len):
        super().__init__()
        data_path = Path(data_path)
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len
        self.type = type

        affinity_path = '/data/affinity2020.csv'
        affinity_data = pd.read_csv(affinity_path, index_col=0)
        affinity = {}
        for _, row in affinity_data.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity

        seq_path = data_path / f'{type}.csv'
        seq_data = pd.read_csv(seq_path)
        smile = {}
        sequence = {}
        idx = {}
        i = 0
        for _, row in seq_data.iterrows():
            idx[i] = row[1]
            smile[row[1]] = row[2]
            sequence[row[1]] = row[3]
            i += 1
        self.smile = smile
        self.sequence = sequence
        self.idx = idx
        assert len(sequence) == len(smile)
        self.len = len(self.sequence)
        self.ligand_info_dict = torch.load("/data/ligand_train2020.pt")
        self.protein_info_dict = torch.load("/data/protein_train2020.pt")
        
    def __getitem__(self, index):
        global device
        id_name = self.idx[index]
        smi = self.smile[id_name]
        seq = self.sequence[id_name]

        smi_encode = torch.tensor(process_smiles(smi,self.ligand_info_dict),device=device).float()
        combined_data = process_protein(seq,self.protein_info_dict)
        ProteinData = namedtuple('ProteinData', ['sequence', 'graph'])
        seq_encode = ProteinData(combined_data["sequence"], combined_data["graph"])
        affinity = torch.tensor(np.array(self.affinity[id_name], dtype=np.float32), device=device)
        return id_name, smi_encode, seq_encode, affinity

    def __len__(self):
        return self.len
    

