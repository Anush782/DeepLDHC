import os
import joblib
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (roc_auc_score, confusion_matrix, accuracy_score, 
                            precision_score, recall_score, f1_score, 
                            matthews_corrcoef, roc_curve, auc)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import RocCurveDisplay
import logging
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV

# Set up color palette for plots
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
BLUE_PALETTE = sns.color_palette("Blues_r")
CMAP = "Blues"

def create_model_folders(base_dir, model_names):
    """Create folder structure for each model"""
    for model_name in model_names:
        model_dir = os.path.join(base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create subdirectories for each model
        subdirs = [
            "models", 
            "optuna_logs", 
            "saved_models", 
            "training_logs",
            "metrics",
            "plots"
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(model_dir, subdir), exist_ok=True)

# Initialize models with their names
MODELS = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
    'LightGBM': LGBMClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'KNeighbors': KNeighborsClassifier(),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'NaiveBayes': GaussianNB()
}

# Set up logging
def setup_logger(name, log_file, level=logging.INFO):
    """Configure logging for training process"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

def calculate_all_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive metrics for binary classification"""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Specificity': recall_score(y_true, y_pred, pos_label=0),
        'F1': f1_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

def plot_confusion_matrix(y_true, y_pred, model_name, path):
    """Plot confusion matrix as heatmap with blue-white palette"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=CMAP, 
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_proba, model_name, path):
    """Plot ROC curve with enhanced styling"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color=BLUE_PALETTE[2], lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()

class ModelFramework:
    def __init__(self, X, y, n_trials=30, n_folds=8, random_state=42):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.random_state = random_state
        self.study = {}
        self.best_models = {}
        self.best_params = {}
        self.cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Create folder structure for each model
        self.BASE_DIR = "Classification_ml_model_results"
        create_model_folders(self.BASE_DIR, MODELS.keys())
        
        # Initialize models
        self.models = MODELS
        
        # Initialize studies
        for model_name in self.models:
            optuna_log_dir = os.path.join(self.BASE_DIR, model_name, "optuna_logs")
            os.makedirs(optuna_log_dir, exist_ok=True)
            
            self.study[model_name] = optuna.create_study(
                direction='maximize',
                study_name=f'{model_name}_study',
                storage=f'sqlite:///{os.path.join(optuna_log_dir, f"{model_name}_optuna.db")}',
                load_if_exists=True
            )
    
    def objective(self, trial, model_name):
        model = clone(self.models[model_name])
        
        if model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
        elif model_name == 'LightGBM':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
        elif model_name == 'AdaBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, log=True),
                'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
            }
        elif model_name == 'SVM':
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            params = {
                'C': trial.suggest_float('C', 0.1, 10, log=True),
                'kernel': kernel,
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'degree': trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3,
                'coef0': trial.suggest_float('coef0', 0.0, 1.0) if kernel in ['poly', 'sigmoid'] else 0.0
            }
        elif model_name == 'LogisticRegression':
            params = {
                'C': trial.suggest_float('C', 0.001, 10, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }
        elif model_name == 'KNeighbors':
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2)  # 1: manhattan, 2: euclidean
            }
        elif model_name == 'GradientBoosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        elif model_name == 'NaiveBayes':
            params = {}  # Naive Bayes typically doesn't have hyperparameters to tune
        
        model.set_params(**params)
        
        # Perform cross-validation
        auc_scores = cross_val_score(
            model, self.X, self.y, 
            cv=self.cv, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return np.mean(auc_scores)
    
    def optimize_models(self):
        for model_name in self.models:
            model_dir = os.path.join(self.BASE_DIR, model_name)
            
            # Setup training logger for this model
            training_logger = setup_logger(
                f'{model_name}_training_logger',
                os.path.join(model_dir, 'training_logs', 'training.log')
            )
            
            training_logger.info(f"Starting optimization for {model_name}")
            
            # Setup Optuna logger for this model
            optuna_logger = setup_logger(
                f'{model_name}_optuna_logger',
                os.path.join(model_dir, 'optuna_logs', 'optuna.log')
            )
            
            def logging_callback(study, trial):
                optuna_logger.info(
                    f"Trial {trial.number} finished with value: {trial.value} "
                    f"and parameters: {trial.params}"
                )
            
            # Run optimization
            self.study[model_name].optimize(
                lambda trial: self.objective(trial, model_name),
                n_trials=self.n_trials,
                callbacks=[logging_callback],
                gc_after_trial=True
            )
            
            # Save best parameters
            self.best_params[model_name] = self.study[model_name].best_params
            training_logger.info(f"Best parameters for {model_name}: {self.best_params[model_name]}")
            
            # Train and save models for each fold
            fold_metrics = []
            for fold, (train_idx, val_idx) in enumerate(self.cv.split(self.X, self.y)):
                X_train, X_val = self.X[train_idx], self.X[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]
                
                # Train model with best parameters
                model = clone(self.models[model_name])
                model.set_params(**self.best_params[model_name])
                model.fit(X_train, y_train)
                
                # Save fold model
                fold_model_path = os.path.join(model_dir, 'models', f'fold_{fold}_model.pkl')
                joblib.dump(model, fold_model_path)
                
                # Evaluate on validation set
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                metrics = calculate_all_metrics(y_val, y_pred, y_proba)
                metrics['Fold'] = fold
                fold_metrics.append(metrics)
                
                # Save plots for this fold
                plot_dir = os.path.join(model_dir, 'plots')
                plot_confusion_matrix(y_val, y_pred, f"{model_name} (Fold {fold})", 
                                    os.path.join(plot_dir, f'fold_{fold}_confusion_matrix.png'))
                plot_roc_curve(y_val, y_proba, f"{model_name} (Fold {fold})", 
                             os.path.join(plot_dir, f'fold_{fold}_roc_curve.png'))
            
            # Save fold metrics to CSV
            metrics_df = pd.DataFrame(fold_metrics)
            metrics_path = os.path.join(model_dir, 'metrics', 'fold_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            
            # Calculate and save average metrics
            avg_metrics = metrics_df.mean().to_dict()
            avg_metrics['Model'] = model_name
            avg_metrics_path = os.path.join(model_dir, 'metrics', 'average_metrics.csv')
            pd.DataFrame([avg_metrics]).to_csv(avg_metrics_path, index=False)
            
            # Train final model on all data
            final_model = clone(self.models[model_name])
            final_model.set_params(**self.best_params[model_name])
            final_model.fit(self.X, self.y)
            self.best_models[model_name] = final_model
            
            # Save the final model
            model_path = os.path.join(model_dir, 'saved_models', 'final_model.pkl')
            joblib.dump(final_model, model_path)
            
            # Generate and save final metrics and plots
            y_pred = final_model.predict(self.X)
            y_proba = final_model.predict_proba(self.X)[:, 1]
            
            # Calculate final metrics
            final_metrics = calculate_all_metrics(self.y, y_pred, y_proba)
            final_metrics_df = pd.DataFrame([final_metrics])
            final_metrics_path = os.path.join(model_dir, 'metrics', 'final_metrics.csv')
            final_metrics_df.to_csv(final_metrics_path, index=False)
            
            # Save final plots
            plot_confusion_matrix(self.y, y_pred, f"{model_name} (Final)", 
                                os.path.join(plot_dir, 'final_confusion_matrix.png'))
            plot_roc_curve(self.y, y_proba, f"{model_name} (Final)", 
                         os.path.join(plot_dir, 'final_roc_curve.png'))
            
            training_logger.info(f"Completed training and evaluation for {model_name}")

# Example usage (uncomment and modify as needed)
X = np.random.rand(100, 50)  # Replace with your 1500x500 data
y = np.random.randint(0, 2, 100)  # Replace with your binary labels

# Example usage
if __name__ == "__main__":
    # Initialize and run the framework
    framework = ModelFramework(X, y, n_trials=30, n_folds=8)
    framework.optimize_models()
