import warnings
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import os
import torch
from torch import tensor

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# Function to calculate molecular fingerprint on GPU if available
def calculate_fingerprint(mol):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        # Convert RDKit ExplicitBitVect to a PyTorch tensor and move to GPU
        fp_tensor = torch.zeros(fp.GetNumBits(), dtype=torch.float32, device=device) # Create a tensor of zeros on the GPU
        for bit in fp.GetOnBits():
            fp_tensor[bit] = 1.0
    return fp_tensor


# Main directory containing the part1 to part7 folders
main_directory = os.path.expanduser(
    '~/Desktop/Anush Karampuri/SMALL_MOLECULE_LIBRARIES/chEMBL/DOWNLOAD-OxgrfN2HR-GMjA4Z-EQdBhXVWbEdhNhe6guQCdZ6WXI=')


# Column name in the CSV file containing SMILES strings
smiles_column = 'smiles'  # Replace 'smiles' with the actual column name


# Iterate over each SMILES ID in the dataframe
for index, row in df.iterrows():
    smiles_id = row[smiles_column]
    print(smiles_id)
    smiles_mol = Chem.MolFromSmiles(smiles_id)
    if smiles_mol is None:
        print(f"Invalid SMILES: {smiles_id}")
        continue
    smiles_fp = calculate_fingerprint(smiles_mol)

    results = []  # List to store results for this SMILES ID

    # Iterate through the part folders (part1 to part7)
    for i in range(1, 8):
        part_folder = os.path.join(main_directory, f'part{i}')
        print(f"Processing folder: {part_folder}")

        # Iterate over each SDF file in the part folder
        for filename in os.listdir(part_folder):
            if filename.endswith('.sdf'):
                sdf_path = os.path.join(part_folder, filename)
                print(f"  Processing SDF file: {filename}")
                suppl = Chem.SDMolSupplier(sdf_path)
                for mol in suppl:
                    if mol is None:
                        continue
                    sdf_fp = calculate_fingerprint(mol)

                    # Calculate Tanimoto similarity using GPU tensors
                    intersection = torch.sum(smiles_fp * sdf_fp)
                    union = torch.sum(smiles_fp + sdf_fp)
                    similarity_score = (intersection / (union - intersection)).cpu().item() if union > intersection else 0.0

                    sdf_smiles = Chem.MolToSmiles(mol)  # Get SMILES from SDF mol
                    results.append({'Reference_SMILES_ID': smiles_id, 'Similar_Molecule_SMILES_ID': sdf_smiles,
                                    'Similarity_Score': similarity_score})

    # Create a DataFrame from the results for this SMILES ID
    results_df = pd.DataFrame(results)

    # Sort by similarity score
    results_df = results_df.sort_values(by='Similarity_Score', ascending=False)

    # Take the top 10%
    top_10_percent = int(len(results_df) * 0.10)
    results_df = results_df.iloc[:top_10_percent]

    print(f"Top 10% similar molecules for SMILES ID '{smiles_id}':")
    print(results_df)
    print("-" * 50)

    # Save the DataFrame to a CSV file immediately
    results_df.to_csv(f"{smiles_id}.csv", index=False)

print("Processing complete.  Results saved to individual CSV files.")



