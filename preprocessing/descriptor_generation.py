import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def normalize_descriptors(df, mol_cols, prot_cols):
    scaler_mol = RobustScaler()
    scaler_prot = RobustScaler()

    df[mol_cols] = scaler_mol.fit_transform(df[mol_cols])
    df[prot_cols] = scaler_prot.fit_transform(df[prot_cols])

    return df, scaler_mol, scaler_prot
