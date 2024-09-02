import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def load_and_process_smiles(file_path):
    
    dataset_smiles = pd.read_csv(file_path)
    
    dataset_smiles = dataset_smiles.groupby('Smiles').mean().reset_index()
    
    return dataset_smiles

def generate_fingerprints(smiles_series):

    molecules = smiles_series.apply(Chem.MolFromSmiles)
    
    fingerprints = molecules.apply(lambda m: AllChem.GetMorganFingerprint(m, radius=3))
    
    return fingerprints

def process_fingerprints(fingerprints):
    
    fingerprint_nonzero = fingerprints.apply(lambda m: m.GetNonzeroElements())
    
    hash_codes = []
    for fingerprint in fingerprint_nonzero:
        for key in fingerprint.keys():
            hash_codes.append(key)
    
    unique_hash_codes = sorted(set(hash_codes))
    
    hash_code_df = pd.DataFrame(unique_hash_codes).reset_index()
    
    fingerprint_matrix = []
    for fingerprint in fingerprint_nonzero:
        fingerprint_row = [0] * len(unique_hash_codes)
        for key, value in fingerprint.items():
            index = hash_code_df[hash_code_df[0] == key]['index'].values[0]
            fingerprint_row[index] = value
        fingerprint_matrix.append(fingerprint_row)
    
    return pd.DataFrame(fingerprint_matrix)

def filter_columns(fingerprint_df, threshold):
    
    zero_sum = (fingerprint_df == 0).astype(int).sum()
    
    filtered_columns = zero_sum[zero_sum < threshold]
    
    return filtered_columns

def main():
    file_path = "Dataset\Smiles.csv"
    threshold = 350
    
    dataset_smiles = load_and_process_smiles(file_path)
    
    fingerprints = generate_fingerprints(dataset_smiles.Smiles)
    
    fingerprint_df = process_fingerprints(fingerprints)

    filtered_columns = filter_columns(fingerprint_df, threshold)
    
    print(len(filtered_columns))

if __name__ == "__main__":
    main()
