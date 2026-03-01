import torch
import torch.nn as nn
from .transformer_encoder import SequenceTransformer
from .descriptor_encoder import DescriptorEncoder

class TransDTAP(nn.Module):
    def __init__(self, smiles_vocab_size, seq_vocab_size,
                 mol_dim, prot_dim,
                 max_smiles_len=200, max_seq_len=1024):

        super().__init__()

        self.smiles_encoder = SequenceTransformer(
            smiles_vocab_size, max_len=max_smiles_len
        )

        self.seq_encoder = SequenceTransformer(
            seq_vocab_size, max_len=max_seq_len
        )

        self.mol_encoder = DescriptorEncoder(mol_dim)
        self.prot_encoder = DescriptorEncoder(prot_dim)

        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, smiles, seq, mol_desc, prot_desc):
        s1 = self.smiles_encoder(smiles)
        s2 = self.seq_encoder(seq)
        d1 = self.mol_encoder(mol_desc)
        d2 = self.prot_encoder(prot_desc)

        fused = torch.cat([s1, s2, d1, d2], dim=1)
        return self.regressor(fused).squeeze(1)
