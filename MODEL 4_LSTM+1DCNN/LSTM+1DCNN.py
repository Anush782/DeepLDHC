# ======================
# IMPORTS & SETUP
# ======================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            roc_curve, auc)
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

import tensorflow as tf

# Set the device to CPU
tf.config.set_visible_devices([], 'GPU')


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ======================
# DIRECTORY SETUP
# ======================
def create_directory_structure():
    base_dir = "cnn_lstm_ablation_study"
    subdirs = [
        "models",
        "plots/optimization",
        "plots/performance",
        "tables/cv_results",
        "tables/optimization",
        "training_logs"
    ]
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        for subdir in subdirs:
            os.makedirs(os.path.join(base_dir, subdir))
    
    return base_dir

output_dir = create_directory_structure()

# ======================
# MODEL ARCHITECTURE (1D-CNN + LSTM ONLY)
# ======================
def create_model(hyperparams, input_shape):
    input_layer = tf.keras.Input(shape=(input_shape,), name='Input_Layer')
    reshaped_input = layers.Reshape((input_shape, 1))(input_layer)
    
    # CNN Layer
    cnn_layer = layers.Conv1D(
        filters=hyperparams['cnn_filters'],
        kernel_size=hyperparams['cnn_kernel'],
        activation='relu',
        kernel_initializer='he_normal'
    )(reshaped_input)
    cnn_layer = layers.BatchNormalization()(cnn_layer)
    cnn_layer = layers.MaxPooling1D(pool_size=2)(cnn_layer)
    cnn_layer = layers.Dropout(hyperparams['dropout_rate'])(cnn_layer)
    
    # LSTM Layer (direct to output)
    lstm_layer = layers.LSTM(
        hyperparams['lstm_units'],
        return_sequences=False,
        kernel_initializer='he_normal',
        recurrent_dropout=hyperparams['recurrent_dropout']
    )(cnn_layer)
    lstm_layer = layers.BatchNormalization()(lstm_layer)
    lstm_layer = layers.Dropout(hyperparams['dropout_rate'])(lstm_layer)
    
    # Output layer for binary classification
    output_layer = layers.Dense(1, activation='sigmoid', name='Output')(lstm_layer)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    optimizer = optimizers.Adam(
        learning_rate=hyperparams['learning_rate'],
        clipvalue=0.5  # Gradient clipping for stability
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

# ======================
# OPTUNA OPTIMIZATION
# ======================
def objective(trial):
    hyperparams = {
        'cnn_filters': trial.suggest_categorical('cnn_filters', [32, 64, 128]),
        'cnn_kernel': trial.suggest_int('cnn_kernel', 3, 7, step=2),
        'lstm_units': trial.suggest_categorical('lstm_units', [64, 128, 256]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
        'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.1, 0.3),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256])
    }
    
    model = create_model(hyperparams, X_train.shape[1])
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=hyperparams['batch_size'],
        callbacks=[early_stopping],
        verbose=1
    )
    
    val_loss = min(history.history['val_loss'])
    return val_loss


# Function to save optimization results
def save_optimization_results(study, output_dir):
    # Save study object
    joblib.dump(study, os.path.join(output_dir, "models", "optuna_study.pkl"))
    
    # Save optimization history as CSV
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(output_dir, "tables/optimization", "optimization_history.csv"), index=False)
    
    # Save best parameters
    with open(os.path.join(output_dir, "models", "best_params.json"), 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    # Plot optimization history
    fig = plot_optimization_history(study)
    fig.write_image(os.path.join(output_dir, "plots/optimization", "optimization_history.png"))
    
    # Plot parameter importance
    fig = plot_param_importances(study)
    fig.write_image(os.path.join(output_dir, "plots/optimization", "parameter_importance.png"))

# Function to evaluate model
def evaluate_model(model, X_test, y_test, output_dir, fold=None):
    # Make predictions
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if fold is not None:
        plt.title(f'Confusion Matrix (Fold {fold})')
        cm_path = os.path.join(output_dir, "plots/performance", f"confusion_matrix_fold_{fold}.png")
        metrics_path = os.path.join(output_dir, "tables/cv_results", f"metrics_fold_{fold}.csv")
    else:
        plt.title('Confusion Matrix (Final Model)')
        cm_path = os.path.join(output_dir, "plots/performance", "confusion_matrix_final.png")
        metrics_path = os.path.join(output_dir, "tables", "final_metrics.csv")
    
    plt.savefig(cm_path)
    plt.close()
    
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    if fold is not None:
        plt.title(f'ROC Curve (Fold {fold})')
        roc_path = os.path.join(output_dir, "plots/performance", f"roc_curve_fold_{fold}.png")
    else:
        plt.title('ROC Curve (Final Model)')
        roc_path = os.path.join(output_dir, "plots/performance", "roc_curve_final.png")
    
    plt.savefig(roc_path)
    plt.close()
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    
    return metrics


def cross_validation(X, y, best_params, output_dir, n_splits=8):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nTraining Fold {fold + 1}/{n_splits}")
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model = create_model(best_params, X_train_fold.shape[1])
        
        callbacks_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ModelCheckpoint(
                os.path.join(output_dir, "models", f"best_model_fold_{fold}.keras"),  # Changed to .keras
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=50,
            batch_size=best_params['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        fold_metrics = evaluate_model(model, X_val_fold, y_val_fold, output_dir, fold)
        fold_metrics['fold'] = fold
        cv_results.append(fold_metrics)
        
        pd.DataFrame(history.history).to_csv(
            os.path.join(output_dir, "training_logs", f"training_history_fold_{fold}.csv"),
            index=False
        )
    
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(output_dir, "tables/cv_results", "cross_validation_results.csv"), index=False)
    return cv_df


# Main function for ablation study
def main_ablation(X, y):
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    
    save_optimization_results(study, output_dir)
    print("\nBest hyperparameters:")
    print(study.best_params)
    
    print("\nStarting 8-Fold Cross-Validation...")
    cv_results = cross_validation(X_train, y_train, study.best_params, output_dir)
    
    print("\nTraining Final Model...")
    final_model = create_model(study.best_params, X_train.shape[1])
    
    final_callbacks = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            os.path.join(output_dir, "models", "final_model.keras"),
            save_best_only=True
        ),
        callbacks.CSVLogger(
            os.path.join(output_dir, "training_logs", "final_training_log.csv")
        )
    ]
    
    final_model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=150,
        batch_size=study.best_params['batch_size'],
        callbacks=final_callbacks,
        verbose=1
    )
    
    print("\nFinal Evaluation on Test Set:")
    final_metrics = evaluate_model(final_model, X_test, y_test, output_dir)
    print("\nTest Set Metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    final_model.save(os.path.join(output_dir, "models", "final_model.keras"))
    print("\nModel saved successfully.")
    return final_model, study.best_params, cv_results
    
# Model loading and prediction snippet
def load_and_predict(model_path, new_data):
    """
    Load a saved model and make predictions on new data
    
    Args:
        model_path: Path to the saved model
        new_data: New data for prediction (numpy array)
        
    Returns:
        Tuple of (class predictions, probability predictions)
    """
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(new_data).flatten()
    predictions = (probabilities > 0.5).astype(int)
    return predictions, probabilities


# Example usage:
X = np.random.rand(1500, 500)  # Your 1500x500 data
y = np.random.randint(0, 2, 1500)  # Your binary labels


final_model, best_params, cv_results = main_ablation(X, y)


# Example usage of loading and prediction:
predictions, probabilities = load_and_predict("cnn_lstm_ablation_study/models/final_model.keras", X_test)  # or any new data in the same format as training data



