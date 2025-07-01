import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, roc_curve)
from keras.models import Sequential, save_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop, SGD, Nadam
from keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import json
import joblib
from datetime import datetime

# Setup directory structure
def setup_dirs():
    base_dir = "hyperopt_results"
    dirs = {
        'base': base_dir,
        'models': os.path.join(base_dir, "trained_models"),
        'plots': os.path.join(base_dir, "visualizations"),
        'params': os.path.join(base_dir, "hyperparameters"),
        'metrics': os.path.join(base_dir, "performance_metrics"),
        'optimization': os.path.join(base_dir, "optimization_data")
    }
    
    # Clear previous results if needed
    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(base_dir)
    
    # Create fresh directory structure
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

dirs = setup_dirs()

# Generate synthetic data with more realistic characteristics
def generate_synthetic_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        class_sep=1.2,
        flip_y=0.05,
        random_state=42
    )
    
    # Add some feature names for interpretability
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    return X, y, feature_names

X, y, feature_names = generate_synthetic_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the model architecture with more flexibility
def create_model(params, input_shape):
    model = Sequential()
    
    # First hidden layer
    model.add(Dense(
        units=int(params['units1']),
        activation=params['activation1'],
        input_dim=input_shape,
        kernel_initializer=params.get('kernel_initializer', 'glorot_uniform')
    ))
    if params['dropout1'] > 0:
        model.add(Dropout(params['dropout1']))
    
    # Second hidden layer
    model.add(Dense(
        units=int(params['units2']),
        activation=params['activation2'],
        kernel_initializer=params.get('kernel_initializer', 'glorot_uniform')
    ))
    if params['dropout2'] > 0:
        model.add(Dropout(params['dropout2']))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Optimizer configuration
    optimizer_map = {
        'adam': Adam,
        'rmsprop': RMSprop,
        'sgd': SGD,
        'nadam': Nadam
    }
    
    optimizer = optimizer_map[params['optimizer']](
        learning_rate=params['learning_rate'],
        **params.get('optimizer_params', {})
    )
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', 'AUC']
    )
    
    return model

# Enhanced hyperparameter space
space = {
    'units1': hp.quniform('units1', 32, 256, 32),
    'units2': hp.quniform('units2', 16, 128, 16),
    'activation1': hp.choice('activation1', ['relu', 'tanh', 'elu']),
    'activation2': hp.choice('activation2', ['relu', 'tanh', 'elu']),
    'optimizer': hp.choice('optimizer', ['adam', 'rmsprop', 'sgd', 'nadam']),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
    'dropout1': hp.uniform('dropout1', 0.0, 0.5),
    'dropout2': hp.uniform('dropout2', 0.0, 0.5),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
    'epochs': hp.choice('epochs', [50, 100, 150]),
    'kernel_initializer': hp.choice('kernel_initializer', ['glorot_uniform', 'he_normal']),
    'optimizer_params': hp.choice('optimizer_params', [
        {},
        {'beta_1': 0.9, 'beta_2': 0.999},
        {'momentum': 0.9},
        {'clipvalue': 0.5}
    ])
}

# Enhanced objective function with early stopping and validation
def objective(params):
    try:
        model = create_model(params, X_train.shape[1])
        
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=int(params['epochs']),
            batch_size=int(params['batch_size']),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate on test set
        y_pred_prob = model.predict(X_test).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Calculate multiple metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_prob),
            'params': params,
            'epochs_used': len(history.history['loss'])
        }
        
        # Save model temporarily for potential later use
        temp_model_path = os.path.join(dirs['models'], f"temp_model_{len(trials.trials)}.h5")
        save_model(model, temp_model_path)
        
        # We optimize for f1 score (can be changed to any other metric)
        return {'loss': -metrics['f1'], 'status': STATUS_OK, 'metrics': metrics}
    
    except Exception as e:
        print(f"Error in trial: {e}")
        return {'loss': 0, 'status': STATUS_FAIL, 'error': str(e)}

# Run optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials,
    rstate=np.random.RandomState(42)
)

# Save optimization results
def save_optimization_results(trials, best_params):
    # Save all trials data
    trials_df = pd.DataFrame([{
        **trial['result']['metrics'],
        'loss': trial['result']['loss'],
        'trial_number': i
    } for i, trial in enumerate(trials.trials) if 'metrics' in trial['result']])
    
    trials_df.to_csv(os.path.join(dirs['optimization'], 'all_trials.csv'), index=False)
    
    # Save best parameters
    best_trial = trials.best_trial
    with open(os.path.join(dirs['params'], 'best_params.json'), 'w') as f:
        json.dump(best_trial['result']['metrics']['params'], f, indent=2)
    
    # Save best model (copy from temp to permanent)
    best_model_path = os.path.join(dirs['models'], "best_model.h5")
    temp_best_path = os.path.join(dirs['models'], f"temp_model_{best_trial['tid']}.h5")
    if os.path.exists(temp_best_path):
        os.rename(temp_best_path, best_model_path)
    
    # Clean up temp models
    for f in os.listdir(dirs['models']):
        if f.startswith("temp_model"):
            os.remove(os.path.join(dirs['models'], f))
    
    return trials_df, best_trial

trials_df, best_trial = save_optimization_results(trials, best)

# Visualization functions
def plot_optimization_history(trials_df):
    plt.figure(figsize=(12, 6))
    plt.plot(trials_df['trial_number'], -trials_df['loss'], 'o-')
    plt.xlabel('Trial Number')
    plt.ylabel('F1 Score (validation)')
    plt.title('Optimization Progress')
    plt.grid(True)
    plt.savefig(os.path.join(dirs['plots'], 'optimization_progress.png'))
    plt.close()

def plot_metrics_comparison(trials_df):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        axes[i].scatter(trials_df['trial_number'], trials_df[metric], alpha=0.6)
        axes[i].set_title(metric)
        axes[i].set_xlabel('Trial Number')
        axes[i].set_ylabel(metric)
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(dirs['plots'], 'metrics_comparison.png'))
    plt.close()

def plot_confusion_matrix_and_roc(best_model_path, X_test, y_test):
    model = load_model(best_model_path)
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Best Model)')
    plt.savefig(os.path.join(dirs['plots'], 'best_model_confusion_matrix.png'))
    plt.close()
    
    # ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Best Model)')
    plt.legend()
    plt.savefig(os.path.join(dirs['plots'], 'best_model_roc_curve.png'))
    plt.close()

# Generate visualizations
plot_optimization_history(trials_df)
plot_metrics_comparison(trials_df)
if os.path.exists(os.path.join(dirs['models'], "best_model.h5")):
    plot_confusion_matrix_and_roc(os.path.join(dirs['models'], "best_model.h5"), X_test, y_test)

# Print summary
print("\n=== Optimization Summary ===")
print(f"Best F1 Score: {-best_trial['result']['loss']:.4f}")
print(f"Best Parameters:")
print(json.dumps(best_trial['result']['metrics']['params'], indent=2))
print(f"\nAll results saved in: {dirs['base']}")
