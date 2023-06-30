import pandas as pd
import torch
import uuid
from typing import Optional
from torch_geometric.data import Dataset, Data

from tqdm import tqdm
import numpy as np 
import os
from deepchem.feat import MolGraphConvFeaturizer, GraphData
from rdkit import Chem 

def _featurize_mol(
        mol : Chem.rdchem.Mol,
        name : Optional[str] = None
    ) -> Data:
    """
    Featurizes Chem.rdchem.Mol object and returns the torch_geometric.data.Data object.
    """
    featurizer = MolGraphConvFeaturizer(use_edges=True, use_chirality=True)
    featurized = featurizer.featurize(mol)[0]
    print(featurized)
    f = GraphData(node_features=featurized.node_features, edge_index=featurized.edge_index, edge_features=featurized.edge_features)
    
    if name is None:
        name = uuid.uuid4()

    data = f.to_pyg_graph()
    data.name = name
    return data

def featurize_fasta(
        fasta : str,
        name : Optional[str] = None
    ) -> Data:
    """
    Featurizes FASTA string and returns the torch_geometric.data.Data object.
    It convert it internally to rdkit.Chem.rdchem.Mol via rdkit.Chem.rdmolfiles.MolFromFASTA (default configuration 0 Protein, L amino acids).
    """
    mol = Chem.MolFromFASTA(fasta)
    return _featurize_mol(mol, name)

def featurize_smiles(
        smiles : str,
        name : Optional[str] = None
    ) -> Data:
    """
    Featurizes SMILES string and returns the torch_geometric.data.Data object.
    It convert it internally to rdkit.Chem.rdchem.Mol via rdkit.Chem.rdmolfiles.MolFromSmiles.
    """
    mol = Chem.MolFromSmiles(smiles)
    return _featurize_mol(mol, name)
    
    

class CPPDataset(Dataset):
    def __init__(self, root='dataset', _split='train', transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split into raw_dir and processed_dir (processed data). 
        """
        self.split = _split
        super(CPPDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return ['train.csv', 'val.csv', 'test.csv', 'mlcpp2_independent.csv']

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv("{}/{}.csv".format(self.raw_dir, self.split)).reset_index()

        return [f'{self.split}_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass    

    def process(self):
        self.data = pd.read_csv("{}/{}.csv".format(self.raw_dir, self.split)).reset_index()
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            # Featurize molecule
            data = featurize_smiles(row["smiles"], row["name"])
            data.y = self._get_label(row["label"])
            data.smiles = row["smiles"]
            data.name = row["name"]
            torch.save(data, os.path.join(self.processed_dir, f'{self.split}_{index}.pt'))
            

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int32)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'{self.split}_{idx}.pt'))
        return data

def load_dataset_cpp(dataset_dir, split='train'):
    return CPPDataset(root=dataset_dir, _split=split)

    

