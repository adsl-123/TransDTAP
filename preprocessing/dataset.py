import torch
from torch.utils.data import Dataset

class DTADataset(Dataset):
    def __init__(self, df, smiles_tokens, seq_tokens, mol_props, prot_props, targets):
        self.smiles = smiles_tokens
        self.seqs = seq_tokens
        self.mol_props = df[mol_props].values
        self.prot_props = df[prot_props].values
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.smiles[idx], dtype=torch.long),
            torch.tensor(self.seqs[idx], dtype=torch.long),
            torch.tensor(self.mol_props[idx], dtype=torch.float32),
            torch.tensor(self.prot_props[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )
