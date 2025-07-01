#!/usr/bin/env python3
import os
import sys
import logging
import warnings
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import cpu_count

# =============================================
# Configuration and Initial Setup
# =============================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shap_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================
# Library Import with Fallback Handling
# =============================================

def import_required_libraries():
    """Import required libraries with fallback options"""
    global tf, shap
    
    # Try TensorFlow import
    try:
        import tensorflow as tf
        logger.info(f"Successfully imported TensorFlow {tf.__version__}")
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except ImportError as e:
        logger.error(f"TensorFlow import failed: {str(e)}")
        tf = None
    
    # Try SHAP import
    try:
        import shap
        logger.info(f"Successfully imported SHAP {shap.__version__}")
    except ImportError as e:
        logger.error(f"SHAP import failed: {str(e)}")
        shap = None
    
    # Check if we have at least one backend
    if tf is None and shap is None:
        logger.error("Neither TensorFlow nor SHAP could be imported. Exiting.")
        sys.exit(1)

# =============================================
# Configuration Class
# =============================================

class Config:
    MODEL_PATH = os.path.expanduser("~/Desktop/Anush Karampuri/MODEL/BInary_method/dnn_mol_desc_ldh_attempt1/models/final_model.keras")
    OUTPUT_DIR = "shap_analysis_results_test"
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    DATA_DIR = os.path.join(OUTPUT_DIR, "processed_data")
    SAMPLE_SIZE = 100  # Start with smaller sample for testing
    TOP_FEATURES = 20
    RANDOM_STATE = 42
    N_CORES = min(8, cpu_count())  # Conservative core count
    SHAP_BATCH_SIZE = 100  # Process in smaller batches

# =============================================
# Core Functions
# =============================================

def setup_directories():
    """Create output directories"""
    try:
        os.makedirs(Config.PLOTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(Config.PLOTS_DIR, "dependency_plots"), exist_ok=True)
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        logger.info("Created output directories")
    except Exception as e:
        logger.error(f"Failed to create directories: {str(e)}")
        raise

def load_model():
    """Load model with compatibility checks"""
    try:
        if tf is not None:
            model = tf.keras.models.load_model(Config.MODEL_PATH)
            logger.info("Model loaded successfully with TensorFlow backend")
            return model
        else:
            raise ImportError("TensorFlow not available")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

def load_data():
    """Load and prepare data"""
    try:
        # Replace this with your actual data loading code
        df2 = pd.read_excel('processed_data.xlsx')
        ''' make sure the data you give as a test dataset should match the number of features with the training dataset features '''
        test_data = df2.copy()
        smiles = test_data["SMILES"]
        X_test = test_data.drop(columns=["SMILES"])
        descriptor_names = X_test.columns.tolist()
        logger.info(f"Loaded data with {len(X_test)} samples and {len(descriptor_names)} features")
        return X_test, descriptor_names, smiles
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def compute_shap_values(model, X_test, descriptor_names):
    """Compute SHAP values with robust error handling"""
    try:
        logger.info(f"Computing SHAP values for {Config.SAMPLE_SIZE} samples")
        
        # Create background and sample datasets
        background = X_test.sample(min(Config.SAMPLE_SIZE, len(X_test)), 
                                 random_state=Config.RANDOM_STATE)
        sample_data = X_test.sample(min(Config.SAMPLE_SIZE, len(X_test)), 
                                 random_state=Config.RANDOM_STATE)
        
        # Initialize explainer
        if tf is not None:
            explainer = shap.DeepExplainer(model, background.values)
            logger.info("Initialized DeepExplainer with TensorFlow backend")
        else:
            explainer = shap.Explainer(model.predict, background)
            logger.info("Initialized generic Explainer")
        
        # Process in batches to avoid memory issues
        shap_values = []
        n_batches = max(1, len(sample_data) // Config.SHAP_BATCH_SIZE)
        
        for batch in tqdm(np.array_split(sample_data.values, n_batches),
                         desc="Processing SHAP batches"):
            try:
                batch_shap = explainer.shap_values(batch)
                shap_values.append(batch_shap)
            except Exception as e:
                logger.warning(f"Failed on batch: {str(e)}")
                continue
        
        if not shap_values:
            raise ValueError("All SHAP batches failed")
            
        # Combine results
        shap_values = np.concatenate(shap_values, axis=0)
        return shap_values.squeeze(), sample_data
    except Exception as e:
        logger.error(f"SHAP computation failed: {str(e)}")
        raise

def generate_visualizations(shap_values, sample_data, descriptor_names):
    """Generate all visualizations with error handling"""
    try:
        # Calculate feature importance
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': descriptor_names,
            'Mean_SHAP': mean_shap,
            'Std_SHAP': np.std(shap_values, axis=0),
            'Mean_SHAP_Value': shap_values.mean(axis=0)  # Raw mean SHAP (direction matters)
        }).sort_values('Mean_SHAP', ascending=False)
        
        # Save feature importance
        feature_importance.to_csv(
            os.path.join(Config.DATA_DIR, "feature_importance.csv"),
            index=False
        )
        
        # --- Split into Positively & Negatively Impacting Features ---
        # Positive impact (increase model output)
        positive_impact = feature_importance[feature_importance['Mean_SHAP_Value'] > 0]
        positive_impact = positive_impact.sort_values('Mean_SHAP_Value', ascending=False)
        
        # Negative impact (decrease model output)
        negative_impact = feature_importance[feature_importance['Mean_SHAP_Value'] < 0]
        negative_impact = negative_impact.sort_values('Mean_SHAP_Value', ascending=True)  # Most negative first
        
        # Save positive and negative impact features
        positive_impact.to_csv(
            os.path.join(Config.DATA_DIR, "positive_impact_features.csv"),
            index=False
        )
        negative_impact.to_csv(
            os.path.join(Config.DATA_DIR, "negative_impact_features.csv"),
            index=False
        )
        
        # Log top features
        logger.info("\n=== Top Positively Impacting Features ===")
        for i, row in positive_impact.head(10).iterrows():
            logger.info(f"{row['Feature']}: Mean SHAP = {row['Mean_SHAP_Value']:.4f}")
        
        logger.info("\n=== Top Negatively Impacting Features ===")
        for i, row in negative_impact.head(10).iterrows():
            logger.info(f"{row['Feature']}: Mean SHAP = {row['Mean_SHAP_Value']:.4f}")
        
        # Generate summary plots
        generate_summary_plots(shap_values, sample_data, descriptor_names)
        
        # Generate dependency plots for top features
        generate_dependency_plots(shap_values, sample_data, descriptor_names, feature_importance)
        
        return feature_importance
    except Exception as e:
        logger.error(f"Visualization generation failed: {str(e)}")
        raise

def generate_summary_plots(shap_values, sample_data, descriptor_names):
    """Generate summary plots"""
    try:
        # Dot plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            sample_data, 
            feature_names=descriptor_names, 
            plot_type="dot",
            show=False
        )
        plt.title("Global Feature Importance", fontsize=14)
        plt.tight_layout()
        plot_path = os.path.join(Config.PLOTS_DIR, "shap_summary_dot.png")
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved summary dot plot to {plot_path}")
        
        # Bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            sample_data, 
            feature_names=descriptor_names, 
            plot_type="bar",
            show=False
        )
        plt.title("Feature Importance (Absolute Mean SHAP)", fontsize=14)
        plt.tight_layout()
        plot_path = os.path.join(Config.PLOTS_DIR, "shap_summary_bar.png")
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved summary bar plot to {plot_path}")
    except Exception as e:
        logger.error(f"Summary plot generation failed: {str(e)}")
        raise

def generate_dependency_plots(shap_values, sample_data, descriptor_names, feature_importance):
    """Generate dependency plots for top features"""
    try:
        top_features = feature_importance.head(Config.TOP_FEATURES)
        
        for feature in tqdm(top_features['Feature'], desc="Generating dependency plots"):
            try:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    feature,
                    shap_values,
                    sample_data,
                    feature_names=descriptor_names,
                    show=False
                )
                plt.title(f"Dependence Plot for {feature}", fontsize=12)
                plt.tight_layout()
                plot_path = os.path.join(Config.PLOTS_DIR, "dependency_plots", f"{feature}_dependence.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(f"Failed to generate dependence plot for {feature}: {str(e)}")
                continue
    except Exception as e:
        logger.error(f"Dependency plot generation failed: {str(e)}")
        raise

# =============================================
# Main Execution
# =============================================

def main():
    start_time = time.time()
    logger.info("Starting SHAP analysis pipeline")
    
    try:
        # Import libraries with fallback handling
        import_required_libraries()
        
        # Setup directories
        setup_directories()
        
        # Load model and data
        model = load_model()
        X_test, descriptor_names, smiles = load_data()
        
        # Compute SHAP values
        shap_values, sample_data = compute_shap_values(model, X_test, descriptor_names)
        
        # Generate visualizations
        feature_importance = generate_visualizations(shap_values, sample_data, descriptor_names)
        
        # Final report
        logger.info("\n=== Analysis Summary ===")
        logger.info(f"Top 5 important features:")
        for i, row in feature_importance.head(5).iterrows():
            logger.info(f"{i+1}. {row['Feature']}: Mean SHAP = {row['Mean_SHAP']:.4f}")
        
        elapsed = time.time() - start_time
        logger.info(f"\nAnalysis completed successfully in {elapsed:.2f} seconds")
        logger.info(f"Results saved to: {Config.OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
