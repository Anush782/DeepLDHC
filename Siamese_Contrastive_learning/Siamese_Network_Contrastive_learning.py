#WITH HYPERPARAMETER OPTIMIZATION - HYPEROPT


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
import random
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, Trials
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Combine all dataframes
df_all = concatenated_df.copy()

# Extract features
ligand_features = df_all.iloc[:, :500].values
protein_features = df_all.iloc[:, 500:].values

# Normalize features
scaler = StandardScaler()
ligand_features = scaler.fit_transform(ligand_features)
protein_features = scaler.fit_transform(protein_features)

# Convert to tensors
protein_tensor = torch.tensor(protein_features, dtype=torch.float32)
ligand_tensor = torch.tensor(ligand_features, dtype=torch.float32)

# Create embeddings
embeddings = torch.cat((protein_tensor, ligand_tensor), dim=1)
labels = np.concatenate([
    np.zeros(len(high_variance_ldha)),  
    np.ones(len(high_variance_ldhb)),   
    np.full(len(high_variance_ldhc), 2)
])

# ✅ Fix: Pairing function now ensures enough pairs
def create_pairs(embeddings, labels, neg_multiplier=5):
    pairs = []
    pair_labels = []
    n = len(embeddings)
    logging.info("Creating pairs...")

    label_dict = {c: np.where(labels == c)[0] for c in np.unique(labels)}

    for i in tqdm(range(n), desc="Pair Creation"):
        pos_samples = label_dict[labels[i]]
        neg_samples = np.concatenate([label_dict[c] for c in np.unique(labels) if c != labels[i]])

        for j in pos_samples:
            if i < j:
                pairs.append((embeddings[i], embeddings[j]))
                pair_labels.append(torch.tensor([0.0]))

        neg_subset = np.random.choice(neg_samples, size=min(len(neg_samples), neg_multiplier), replace=False)
        for j in neg_subset:
            pairs.append((embeddings[i], embeddings[j]))
            pair_labels.append(torch.tensor([1.0]))

    logging.info(f"Total pairs created: {len(pairs)}")
    logging.info("Pair creation complete.")
    return pairs, pair_labels

# ✅ Fix: Ensure negative sampling is balanced
pairs, pair_labels = create_pairs(embeddings, labels, neg_multiplier=5)

# ✅ Fix: Neural network size optimized
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x1, x2):
        out1 = self.fc(x1)
        out2 = self.fc(x2)
        return out1, out2

# ✅ Fix: Prevents NaNs in loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.epsilon = 1e-6  # Small value to avoid zero gradients

    def forward(self, out1, out2, label):
        distance = torch.nn.functional.pairwise_distance(out1, out2)
        loss = torch.mean((1 - label) * distance ** 2 +
                          label * torch.pow(torch.clamp(self.margin - distance + self.epsilon, min=0.0), 2))
        return loss

# ✅ Hyperparameter Optimization with Hyperopt
def objective(params):
    hidden_size = int(params['hidden_size'])
    lr = params['lr']
    margin = params['margin']

    model = SiameseNetwork(input_dim=embeddings.shape[1], hidden_size=hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = ContrastiveLoss(margin=margin)

    num_epochs = 10
    total_loss = 0

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for (emb1, emb2), label in zip(pairs, pair_labels):
            emb1, emb2, label = emb1.float(), emb2.float(), label.float()
            optimizer.zero_grad()

            out1, out2 = model(emb1, emb2)
            loss = criterion(out1, out2, label)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        total_loss += epoch_loss / len(pairs)

    return total_loss / num_epochs

# Define hyperparameter search space
space = {
    'hidden_size': hp.choice('hidden_size', [128, 256, 512]),
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-3)),
    'margin': hp.uniform('margin', 0.5, 2.0),
}

trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

# Train the model with best params
best_hidden_size = int(best_params['hidden_size'])
best_lr = best_params['lr']
best_margin = best_params['margin']

model = SiameseNetwork(input_dim=embeddings.shape[1], hidden_size=best_hidden_size)
optimizer = optim.Adam(model.parameters(), lr=best_lr)
criterion = ContrastiveLoss(margin=best_margin)


# ✅ Fix: Training with optimized params
num_epochs = 50
for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    total_loss = 0
    for (emb1, emb2), label in zip(pairs, pair_labels):
        emb1, emb2, label = emb1.float(), emb2.float(), label.float()
        optimizer.zero_grad()

        out1, out2 = model(emb1, emb2)
        loss = criterion(out1, out2, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(pairs):.4f}")

# ✅ Extract refined embeddings
model.eval()
with torch.no_grad():
    refined_embeddings = model.fc(embeddings)

# ✅ Save refined dataset
refined_df = pd.DataFrame(refined_embeddings.numpy(), columns=[f"feature_{i}" for i in range(refined_embeddings.shape[1])])
refined_df['isoform'] = concatenated_df1['isoform'].values
refined_df.to_csv("refined_dataset.csv", index=False)
logging.info("Refined dataset saved successfully.")

# ✅ Save the trained model
model_save_path = "trained_siamese_model.pth"
torch.save(model.state_dict(), model_save_path)
logging.info(f"Trained model saved successfully to {model_save_path}.")
