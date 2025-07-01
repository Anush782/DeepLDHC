import os
import joblib
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (r2_score, mean_squared_error, 
                            mean_absolute_error, explained_variance_score)
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'SVR': SVR(),
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'ElasticNet': ElasticNet(random_state=42),
    'KNeighbors': KNeighborsRegressor(),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

def setup_logger(name, log_file, level=logging.INFO):
    """Configure logging for training process"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

def calculate_all_metrics(y_true, y_pred):
    """Calculate comprehensive metrics for regression"""
    mse = mean_squared_error(y_true, y_pred)
    return {
        'R2': r2_score(y_true, y_pred),
        'Adjusted_R2': 1 - (1 - r2_score(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - X.shape[1] - 1),
        'RMSE': np.sqrt(mse),
        'MSE': mse,
        'MAE': mean_absolute_error(y_true, y_pred),
        'Explained_Variance': explained_variance_score(y_true, y_pred),
        'Q2': 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    }

def plot_scatter(y_true, y_pred, model_name, path):
    """Plot predicted vs actual values"""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, color=BLUE_PALETTE[2])
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 
             color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted for {model_name}')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_residuals(y_true, y_pred, model_name, path):
    """Plot residuals"""
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, color=BLUE_PALETTE[2])
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot for {model_name}')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_error_distribution(y_true, y_pred, model_name, path):
    """Plot distribution of errors"""
    errors = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, kde=True, color=BLUE_PALETTE[2])
    plt.xlabel('Prediction Error')
    plt.title(f'Error Distribution for {model_name}')
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
        self.cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Create folder structure for each model
        self.BASE_DIR = "ml_reg_model_results"
        create_model_folders(self.BASE_DIR, MODELS.keys())
        
        # Initialize models
        self.models = MODELS
        
        # Initialize studies
        for model_name in self.models:
            optuna_log_dir = os.path.join(self.BASE_DIR, model_name, "optuna_logs")
            os.makedirs(optuna_log_dir, exist_ok=True)
            
            self.study[model_name] = optuna.create_study(
                direction='minimize',  # Minimizing RMSE
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
                'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential'])
            }
        elif model_name == 'SVR':
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            params = {
                'C': trial.suggest_float('C', 0.1, 10, log=True),
                'kernel': kernel,
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'degree': trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3,
                'coef0': trial.suggest_float('coef0', 0.0, 1.0) if kernel in ['poly', 'sigmoid'] else 0.0
            }

        elif model_name == 'LinearRegression':
            params = {
                 'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                 'positive': trial.suggest_categorical('positive', [True, False])}
            
        elif model_name in ['Ridge', 'Lasso']:
            params = {
                'alpha': trial.suggest_float('alpha', 0.001, 10, log=True),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)}
        elif model_name == 'ElasticNet':
            params = {
                'alpha': trial.suggest_float('alpha', 0.001, 10, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0, 1),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
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
        
        # Create pipeline with standard scaler for models that need it
        if model_name in ['SVR', 'KNeighbors', 'Ridge', 'Lasso', 'ElasticNet']:
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model.set_params(**params))
            ])
        else:
            model.set_params(**params)
        
        # Perform cross-validation (using negative RMSE to maximize)
        rmse_scores = cross_val_score(
            model, self.X, self.y, 
            cv=self.cv, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        return -np.mean(rmse_scores)  # Return positive RMSE
    
    def optimize_models(self):
        for model_name in self.models:
            model_dir = os.path.join(self.BASE_DIR, model_name)
            os.makedirs(os.path.join(model_dir, 'models'), exist_ok=True)
            os.makedirs(os.path.join(model_dir, 'training_logs'), exist_ok=True)
            os.makedirs(os.path.join(model_dir, 'optuna_logs'), exist_ok=True)             
            os.makedirs(os.path.join(model_dir, 'saved_models'), exist_ok=True)
            os.makedirs(os.path.join(model_dir, 'training_logs'), exist_ok=True)
            os.makedirs(os.path.join(model_dir, 'metrics'), exist_ok=True)
            os.makedirs(os.path.join(model_dir, 'plots'), exist_ok=True)
            
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
                
                # Create model with best parameters
                if model_name in ['SVR', 'KNeighbors', 'Ridge', 'Lasso', 'ElasticNet']:
                    model = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', clone(self.models[model_name]).set_params(**self.best_params[model_name]))
                    ])
                else:
                    model = clone(self.models[model_name])
                    model.set_params(**self.best_params[model_name])
                
                model.fit(X_train, y_train)
                
                # Save fold model
                fold_model_path = os.path.join(model_dir, 'models', f'fold_{fold}_model.pkl')
                joblib.dump(model, fold_model_path)
                
                # Evaluate on validation set
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                metrics = calculate_all_metrics(y_val, y_pred)
                metrics['Fold'] = fold
                fold_metrics.append(metrics)
                
                # Save plots for this fold
                plot_dir = os.path.join(model_dir, 'plots')
                plot_scatter(y_val, y_pred, f"{model_name} (Fold {fold})", 
                            os.path.join(plot_dir, f'fold_{fold}_scatter.png'))
                plot_residuals(y_val, y_pred, f"{model_name} (Fold {fold})", 
                             os.path.join(plot_dir, f'fold_{fold}_residuals.png'))
                plot_error_distribution(y_val, y_pred, f"{model_name} (Fold {fold})", 
                                      os.path.join(plot_dir, f'fold_{fold}_error_dist.png'))
            
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
            if model_name in ['SVR', 'KNeighbors', 'Ridge', 'Lasso', 'ElasticNet']:
                final_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', clone(self.models[model_name]).set_params(**self.best_params[model_name]))
                ])
            else:
                final_model = clone(self.models[model_name])
                final_model.set_params(**self.best_params[model_name])
            
            final_model.fit(self.X, self.y)
            self.best_models[model_name] = final_model
            
            # Save the final model
            model_path = os.path.join(model_dir, 'saved_models', 'final_model.pkl')
            joblib.dump(final_model, model_path)
            
            # Generate and save final metrics and plots
            y_pred = final_model.predict(self.X)
            
            # Calculate final metrics
            final_metrics = calculate_all_metrics(self.y, y_pred)
            final_metrics_df = pd.DataFrame([final_metrics])
            final_metrics_path = os.path.join(model_dir, 'metrics', 'final_metrics.csv')
            final_metrics_df.to_csv(final_metrics_path, index=False)
            
            # Save final plots
            plot_scatter(self.y, y_pred, f"{model_name} (Final)", 
                        os.path.join(plot_dir, 'final_scatter.png'))
            plot_residuals(self.y, y_pred, f"{model_name} (Final)", 
                         os.path.join(plot_dir, 'final_residuals.png'))
            plot_error_distribution(self.y, y_pred, f"{model_name} (Final)", 
                                  os.path.join(plot_dir, 'final_error_dist.png'))
            
            training_logger.info(f"Completed training and evaluation for {model_name}")

# Synthetic data generation for testing
def generate_synthetic_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic regression data"""
    np.random.seed(random_state)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Create meaningful coefficients
    coef = np.random.randn(n_features)
    coef[np.random.rand(n_features) < 0.3] = 0  # Make some features irrelevant
    
    # Generate target with noise
    y = X.dot(coef) + noise * np.random.randn(n_samples)
    
    # Add some non-linearities
    y += 0.5 * np.sin(X[:, 0] * 3) + 0.5 * np.exp(X[:, 1])
    
    return X, y

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    X, y = generate_synthetic_data()
    print(f"Generated data with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Initialize and run the framework
    framework = ModelFramework(X, y, n_trials=30, n_folds=8)
    framework.optimize_models()
    
    # Print best models and their metrics
    for model_name, model in framework.best_models.items():
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        print(f"{model_name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
