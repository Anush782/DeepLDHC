import os
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from mol2vec.features import mol2alt_sentence
from gensim.models import Word2Vec
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.data import Data
import time
import gc
from tqdm import tqdm
import psutil
import warnings
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages


def init_mol2vec_worker(MODEL_PATH):
    """Initialize worker with Mol2Vec model."""
    global mol2vec_model, mol2vec_keys
    mol2vec_model = Word2Vec.load(MODEL_PATH)
    mol2vec_keys = set(mol2vec_model.wv.key_to_index.keys())

def process_single_mol2vec(smiles):
    """Process a single SMILES to Mol2Vec features."""
    mol = MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return np.zeros(100)
    
    sent = mol2alt_sentence(mol, 1)
    sent_vec = []
    for word in sent:
        if str(word) in mol2vec_keys:
            sent_vec.append(mol2vec_model.wv[str(word)])
    
    return np.sum(sent_vec, axis=0) if sent_vec else np.zeros(100)


def init_gnn_worker():
    """Initialize worker with GNN model."""
    global gnn_model
    gnn_model = EfficientGNN()
    
def process_single_gcn(smiles):
    """Process a single SMILES with GNN."""
    try:
        graph = smiles_to_graph(smiles)
        graph.to(gnn_model.device)
        emb = gnn_model(graph.x, graph.edge_index).cpu().numpy()
        return emb
    except:
        return np.zeros(400)



# Set multiprocessing start method to 'spawn' at the very beginning
if __name__ == '__main__':
    set_start_method('spawn', force=True)

# =============================================
# 0. Configuration
# =============================================
class Config:
    # Input/Output
    INPUT_DIR = "~/Desktop/Anush Karampuri/MODEL/Binary_classification_approach/Virtual Screening/test"  
    OUTPUT_DIR = "./test_pp"  
    MODEL_PATH = "final_model.keras"  
    MOL2VEC_MODEL = "model_300dim.pkl"  
    CHECKPOINT_FILE = os.path.expanduser("~/checkpoint.pkl")  # To resume processing

    # Processing
    BATCH_SIZE = 10000  # Reduced from 50k to 10k for memory safety
    MOL2VEC_BATCH_SIZE = 5000  # Smaller batch for memory-intensive mol2vec
    GCN_BATCH_SIZE = 2000  # Smaller batch for GCN processing
    PREDICTION_THRESHOLD = 0.7  

    # System
    USE_GPU_FOR_GCN = True  
    FORCE_CPU_FOR_PREDICTION = True  
    MAX_MEMORY_USAGE = 0.9  # 90% of available memory
    SAFE_MEMORY_THRESHOLD = 0.8  # Pause if memory exceeds this
    NUM_WORKERS = max(1, cpu_count() - 4)  # Use all cores except 4 for system stability

# Initialize configuration
config = Config()
config.INPUT_DIR = os.path.expanduser(config.INPUT_DIR)
config.MODEL_PATH = os.path.expanduser(config.MODEL_PATH)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# =============================================
# 1. Memory Management Utilities
# =============================================
def memory_safe():
    """Check if memory usage is within safe limits."""
    mem = psutil.virtual_memory()
    return mem.percent/100 < config.SAFE_MEMORY_THRESHOLD

def clear_memory():
    """Aggressive memory clearing."""
    gc.collect()
    torch.cuda.empty_cache()
    tf.keras.backend.clear_session()

def wait_for_memory():
    """Pause execution until memory is available."""
    while not memory_safe():
        print("Memory usage high, pausing for 60 seconds...")
        time.sleep(60)
        clear_memory()

# =============================================
# 2. Optimized Feature Generation (Parallelized)
# =============================================
def process_single_mol2vec(smiles, MODEL_PATH):
    """Process a single SMILES to Mol2Vec features."""
    if not hasattr(process_single_mol2vec, 'model'):
        process_single_mol2vec.model = Word2Vec.load(MODEL_PATH)
        process_single_mol2vec.keys = set(process_single_mol2vec.model.wv.key_to_index.keys())
    
    mol = MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return np.zeros(100)
    
    sent = mol2alt_sentence(mol, 1)
    sent_vec = []
    for word in sent:
        if str(word) in process_single_mol2vec.keys:
            sent_vec.append(process_single_mol2vec.model.wv[str(word)])
    
    return np.sum(sent_vec, axis=0) if sent_vec else np.zeros(100)



def compute_mol2vec_features_parallel(df_batch):
    """Compute Mol2Vec embeddings with parallel processing."""
    smiles_list = df_batch["smiles"].tolist()
    
    # Create partial function that doesn't require additional arguments
    with Pool(config.NUM_WORKERS, initializer=init_mol2vec_worker, initargs=(config.MOL2VEC_MODEL,)) as pool:
        results = list(tqdm(pool.imap(process_single_mol2vec, smiles_list), 
                      total=len(smiles_list), 
                      desc="Mol2Vec Processing"))
    
    mol2vec_emb = np.array(results)
    return pd.DataFrame(mol2vec_emb, columns=[f"mol2vec_{i+1}" for i in range(100)])
    
    
    

def init_mol2vec_worker(MODEL_PATH):
    """Initialize worker with Mol2Vec model."""
    global mol2vec_model, mol2vec_keys
    mol2vec_model = Word2Vec.load(MODEL_PATH)
    mol2vec_keys = set(mol2vec_model.wv.key_to_index.keys())

def process_single_mol2vec_wrapper(smiles):
    """Wrapper function for Mol2Vec processing."""
    return process_single_mol2vec(smiles, config.MOL2VEC_MODEL)

def smiles_to_graph(smiles):
    """Convert SMILES to graph representation (implementation not shown)"""
    # Your existing implementation here
    pass

class EfficientGNN(nn.Module):
    """Memory-optimized GCN model with batch processing."""
    def __init__(self, input_dim=1, output_dim=400):
        super(EfficientGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, output_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU_FOR_GCN else "cpu")
        self.to(self.device)
        self.eval()  # Set to evaluation mode
    
    def forward(self, x, edge_index):
        with torch.no_grad():  # Disable gradient calculation
            x = torch.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return x.mean(dim=0)

def init_gnn_worker():
    """Initialize worker with GNN model."""
    global gnn_model
    gnn_model = EfficientGNN()
    
def process_single_gcn(smiles):
    """Process a single SMILES with GNN."""
    try:
        graph = smiles_to_graph(smiles)
        graph.to(gnn_model.device)
        emb = gnn_model(graph.x, graph.edge_index).cpu().numpy()
        return emb
    except:
        return np.zeros(400)



def compute_gcn_features_parallel(df_batch):
    """Compute GCN embeddings with parallel processing."""
    smiles_list = df_batch["smiles"].tolist()
    
    with Pool(config.NUM_WORKERS, initializer=init_gnn_worker) as pool:
        embeddings = list(tqdm(pool.imap(process_single_gcn, smiles_list),
                             total=len(smiles_list),
                             desc="GCN Processing"))
    
    return pd.DataFrame(embeddings, columns=[f"gcn_{i+1}" for i in range(400)])



# =============================================
# 3. Prediction Pipeline with Checkpointing
# =============================================
def load_model():
    """Load the trained Keras model with memory optimization."""
    if config.FORCE_CPU_FOR_PREDICTION:
        tf.config.set_visible_devices([], 'GPU')
    
    # Load model with custom objects if needed
    try:
        model = tf.keras.models.load_model(config.MODEL_PATH)
    except:
        # If model loading fails, try loading weights only
        model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
    
    return model

def predict_batch(model, features_df):
    """Predict probabilities in memory-efficient batches."""
    X = features_df.drop("smiles", axis=1).values
    batch_size = 8192  # Optimal batch size for prediction
    predictions = []
    
    for i in tqdm(range(0, len(X), batch_size), 
                  desc="Making Predictions", 
                  total=len(X)//batch_size + 1):
        batch = X[i:i+batch_size]
        predictions.extend(model.predict(batch, verbose=0).flatten())
        if not memory_safe():
            clear_memory()
    
    return np.array(predictions)

# =============================================
# 4. Main Pipeline with Checkpointing
# =============================================
def save_checkpoint(file_path, batch_idx, global_stats):
    """Save current processing state."""
    checkpoint = {
        'file_path': file_path,
        'batch_idx': batch_idx,
        'global_stats': global_stats
    }
    pd.to_pickle(checkpoint, config.CHECKPOINT_FILE)

def load_checkpoint():
    """Load processing state if available."""
    if os.path.exists(config.CHECKPOINT_FILE):
        return pd.read_pickle(config.CHECKPOINT_FILE)
    return None

def process_file(file_path, model, global_stats, start_batch=0):
    """Process a single input file with checkpointing."""
    # Load data
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, chunksize=config.BATCH_SIZE)
    else:
        df = pd.read_excel(file_path)
        df = [df.iloc[i:i+config.BATCH_SIZE] for i in range(0, len(df), config.BATCH_SIZE)]
    
    print(f"\nProcessing {os.path.basename(file_path)} in batches...")
    
    # Process batches with overall progress bar
    total_batches = len(list(pd.read_csv(file_path, chunksize=config.BATCH_SIZE))) if file_path.endswith(".csv") else len(df)
    pbar = tqdm(total=total_batches, desc=f"Processing {os.path.basename(file_path)}")
    
    for batch_num, df_batch in enumerate(df):
        if batch_num < start_batch:
            pbar.update(1)
            continue  # Skip already processed batches
        
        wait_for_memory()
        
        # Generate features in parallel
        mol2vec_df = compute_mol2vec_features_parallel(df_batch)
        gcn_df = compute_gcn_features_parallel(df_batch)
        features_df = pd.concat([df_batch["smiles"].reset_index(drop=True), 
                                gcn_df.reset_index(drop=True), 
                                mol2vec_df.reset_index(drop=True)], axis=1)
        
        # Predict
        probabilities = predict_batch(model, features_df)
        predictions = (probabilities > config.PREDICTION_THRESHOLD).astype(int)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            "smiles": df_batch["smiles"],
            "probability": probabilities,
            "predicted_class": predictions,
            "mapped_class": np.where(predictions == 1, "LDHC", "LDHA_B")
        })
        
        # Filter results based on probability cutoff before saving
        filtered_df = results_df[
            (results_df["probability"] <= 0.10) | 
            (results_df["probability"] >= 0.90)
        ]
        
        # Only save if there are results meeting the cutoff
        if len(filtered_df) > 0:
            output_path = os.path.join(
                config.OUTPUT_DIR,
                f"{os.path.splitext(os.path.basename(file_path))[0]}_batch_{batch_num}.csv"
            )
            filtered_df.to_csv(output_path, index=False)
        
        # Update stats (using original predictions before filtering)
        global_stats["total_molecules"] += len(df_batch)
        global_stats["ldhc_molecules"] += predictions.sum()
        
        # Save checkpoint
        save_checkpoint(file_path, batch_num + 1, global_stats)
        
        # Clear memory and update progress
        del mol2vec_df, gcn_df, features_df, results_df, filtered_df
        clear_memory()
        pbar.update(1)
    
    pbar.close()
    return global_stats

# =============================================
# 5. Execution with Resume Capability
# =============================================
if __name__ == "__main__":
    # Initialize
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint['file_path']} batch {checkpoint['batch_idx']}")
        global_stats = checkpoint['global_stats']
        start_file = checkpoint['file_path']
        start_batch = checkpoint['batch_idx']
    else:
        global_stats = {"total_molecules": 0, "ldhc_molecules": 0}
        start_file = None
        start_batch = 0
    
    # Load model
    model = load_model()
    
    # Find input files
    input_files = []
    for f in os.listdir(config.INPUT_DIR):
        if f.endswith((".csv", ".xlsx", ".xls")):
            input_files.append(os.path.join(config.INPUT_DIR, f))
    
    if not input_files:
        raise FileNotFoundError(f"No CSV/Excel files found in {config.INPUT_DIR}")
    
    print(f"Found {len(input_files)} input files")
    
    # Process files with progress bar
    processing_started = start_file is None
    with tqdm(input_files, desc="Processing Files") as file_pbar:
        for file_path in file_pbar:
            file_pbar.set_postfix(file=os.path.basename(file_path))
            
            if not processing_started:
                if file_path == start_file:
                    processing_started = True
                else:
                    file_pbar.update(1)
                    continue
            
            try:
                global_stats = process_file(file_path, model, global_stats, start_batch)
                start_batch = 0  # Reset after first file
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                print("Saving checkpoint before exiting...")
                save_checkpoint(file_path, start_batch, global_stats)
                raise
    
    # Final summary
    if os.path.exists(config.CHECKPOINT_FILE):
        os.remove(config.CHECKPOINT_FILE)
    
    summary = {
        "Total Molecules Screened": global_stats["total_molecules"],
        "LDHA_B Predictions": global_stats["total_molecules"] - global_stats["ldhc_molecules"],
        "LDHC Predictions": global_stats["ldhc_molecules"],
        "LDHC Hit Rate (%)": round(100 * global_stats["ldhc_molecules"] / global_stats["total_molecules"], 2)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(config.OUTPUT_DIR, "screening_summary.csv"), index=False)
    
    print("\n=== Virtual Screening Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    
    print(f"\nAll results saved to: {os.path.abspath(config.OUTPUT_DIR)}")
