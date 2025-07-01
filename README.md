# <img src="https://img.icons8.com/color/48/000000/test-tube.png"/> DeepLDHC  
## *Deep learning and virtual screening framework for LDHC inhibitor identification and selectivity profiling*  
### <img src="https://img.icons8.com/fluency/24/000000/artificial-intelligence.png"/> AI-Driven Drug Discovery |  Hybrid ML/DL Approach |  Non-Hormonal Contraception Target  

---

# <img src="https://img.icons8.com/color/30/000000/installer.png"/> Installation  
```bash
pip install -r requirements.txt
```


## Key Dependencies:
## <img src="https://img.icons8.com/color/14/000000/tensorflow.png"/> TensorFlow 2.14
## <img src="https://img.icons8.com/color/14/000000/pytorch.png"/> PyTorch 2.0.1
## <img src="https://img.icons8.com/color/14/000000/python.png"/> Python 3.12+

# <img src="https://img.icons8.com/color/30/000000/folder-invoices.png"/> Repository Structure

├── 📂 scripts/  
│   ├── 🐍 classification.py        → RF/SVM/XGBoost  
│   ├── 📈 regression.py            → Linear/Lasso/ElasticNet  
│   ├── 🧠 neural_networks/         
│   │   ├── 🔄 cnn_lstm.py          ← Hybrid Architecture  
│   │   └── 🎯 dnn_classifier.py    
│   └── ⚙️ hyperparameter_tuning/   ← Optuna Trials  
├── 📂 outputs/  
│   ├── 📊 classification_results/  ← ROC/Confusion Mats  
│   ├── 📉 regression_results/      ← R²/RMSE Plots  
│   └── 📜 nn_training_logs/        ← History CSV  
├── 🗃️ data/  
│   ├── � synthetic_data.csv        ⚠️ Replace with real data  
│   └── 📝 README.md                ← Data Specifications  

# <img src="https://img.icons8.com/color/30/000000/rocket.png"/> Quick Start
1. <img src="https://img.icons8.com/color/20/000000/data-configuration.png"/> Data Preparation
```   
df = pd.read_csv("your_data.csv")  # Shape: (samples, features)
labels = pd.read_csv("your_labels.csv") 
```
# 🚀 How to Use the Scripts
2. Replace Synthetic Data

    * Each script has a section for data loading.
    * Replace the synthetic data snippet with your dataset:
  
   Example: Replace this with your data
   ```
   df = pd.read_csv("data/synthetic_data.csv")  # Remove this  
   df = pd.read_csv("your_dataset.csv")           # Use your file
   ```
3. <img src="https://img.icons8.com/color/20/000000/console.png"/> Execute Scripts
   ```
   # For classification:
   python scripts/classification.py --data_path your_data.csv
   # For neural networks:
   python scripts/neural_networks/cnn_lstm.py --epochs 150
   ```

4. <img src="https://img.icons8.com/color/20/000000/tuning-fork.png"/> Hyperparameter Tuning
   Modify objective() in any Optuna script:
   ```
   def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'units': trial.suggest_categorical('units', [64, 128, 256])
    }
   ```
   
# <img src="https://img.icons8.com/color/30/000000/architecture.png"/> Model Architecture
Component	Key Features
Component Details:
1. INITIALIZATION:
   - Sets random seeds (42)
   - Creates directory structure
   - Disables GPU (optional)

2. DATA LOADING:
   - Receives X,y (placeholder for real data)
   - Performs train-test split (80-20)
   - Stratified sampling for class balance

3. MODEL DEFINITION:
   - 1D-CNN → LSTM → DNN architecture
   - BatchNorm/Dropout layers
   - Custom metrics (AUC, Precision, Recall)

4. HYPERPARAMETER OPTIMIZATION:
   - Optuna searches over:
     • CNN filters/kernel size
     • LSTM units
     • Learning rate (log scale)
     • Batch size
   - Early stopping implemented

5. CROSS-VALIDATION:
   - 8-fold stratified CV
   - Saves per-fold:
     • Model checkpoints
     • Training histories
     • Evaluation metrics

6. FINAL TRAINING:
   - Trains on full 80% training set
   - 150 epochs with:
     • Model checkpointing
     • CSV logger
     • Early stopping

7. PERSISTENCE:
   - Saves:
     • Final model (.keras)
     • Best parameters (JSON)
     • Optimization history (CSV)
     • Visualizations (PNG)

8. DEPLOYMENT:
   - load_and_predict() utility:
     • Loads saved model
     • Makes batch predictions
     • Returns class/probability

<img src="https://img.icons8.com/color/30/000000/experimental-chemistry.png"/> Expected Outputs

Classification:
<img src="https://img.icons8.com/color/20/000000/roc-curve.png"/> ROC Curves | <img src="https://img.icons8.com/color/20/000000/confusion-matrix.png"/> Confusion Matrices

Regression:
<img src="https://img.icons8.com/color/20/000000/line-chart.png"/> Residual Plots | <img src="https://img.icons8.com/color/20/000000/r2.png"/> R² Scores

Neural Nets:
<img src="https://img.icons8.com/color/20/000000/training.png"/> Training Logs | <img src="https://img.icons8.com/color/20/000000/model.png"/> Saved Weights


# <img src="https://img.icons8.com/color/30/000000/mit-license.png"/> License

MIT License - Free for academic/commercial use with attribution

# <img src="https://img.icons8.com/color/20/000000/github.png"/> Contribute:

```
git clone https://github.com/yourusername/DeepLDHC.git  
# Submit PRs to dev branch
```

# <img src="https://img.icons8.com/color/20/000000/bug.png"/> Report Issues:
## *GitHub Issues or enhancements, open a GitHub Issue or submit a Pull Request*

---

# <img src="https://img.icons8.com/color/30/000000/contacts.png"/> Contact  
**Anush Karampuri**  
📧 [anush.karampuri1@gmail.com](mailto:anush.karampuri1@gmail.com)   
💼 [LinkedIn](#) *  *  

*For collaboration or queries about this repository*  
