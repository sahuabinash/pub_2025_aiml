"""
PyTorch tabular example:
- binary classification
- high-cardinality categorical features with frequency-capping + OOV
- end-to-end training loop on synthetic data
- inference transform + save/load
"""

import math
import random
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Config / Defaults (change to your dataset)
# -----------------------------
VOCAB_CONFIG = {
    # feature_name: keep_top_k (None means keep all seen)
    "user_id": 100_000,     # keep top 100k frequent users, rest -> OOV
    "merchant_id": 50_000,  # keep top 50k merchants
    "mcc_code": 1_000       # keep top 1k mccs (usually small)
}
NUMERIC_FEATURES = ["txn_amount", "hour_of_day"]  # example numeric features
EMBEDDING_MAX_DIM = 50
EMBEDDING_DIM_HEURISTIC_POWER = 0.25  # embedding_dim = min(max_dim, int(cardinality**power))
BATCH_SIZE = 1024
LR = 1e-3
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Option: if True, use hashing trick instead of explicit vocab for very large cardinalities
USE_HASHING = False
HASH_BUCKETS = {
    # if using hashing, each feature maps to fixed number buckets
    "user_id": 200_000,
    "merchant_id": 100_000,
    "mcc_code": 2_048
}

# -----------------------------
# Utilities: vocab building + encoding
# -----------------------------
def build_vocab_from_data(rows: List[Dict[str, Any]], vocab_config: Dict[str, int]):
    """
    rows: list of dicts with keys for categorical features
    vocab_config: feature_name -> keep_top_k (int or None)
    returns: dict feature_name -> {value: idx}, plus an 'OOV' reserved index 0
    """
    counters = defaultdict(Counter)
    for row in rows:
        for feat in vocab_config.keys():
            v = row.get(feat)
            if v is None:
                continue
            counters[feat][v] += 1

    vocab = {}
    for feat, keep_top_k in vocab_config.items():
        most_common = counters[feat].most_common(keep_top_k) if keep_top_k else counters[feat].most_common()
        # reserve 0 for OOV / padding
        mapping = {"__OOV__": 0}
        idx = 1
        for value, _count in most_common:
            mapping[value] = idx
            idx += 1
        vocab[feat] = mapping
        # store cardinality as len(mapping) (including OOV)
    return vocab

def apply_hashing(value: Any, buckets: int) -> int:
    # simple deterministic hash -> bucket index (reserve 0 for OOV if you want; here we map to 0..buckets-1)
    return (hash(value) % buckets)

def encode_row(row: Dict[str, Any], vocab: Dict[str, Dict[Any,int]], numeric_feats: List[str],
               use_hashing: bool=False, hash_buckets: Dict[str,int]=None) -> Tuple[Dict[str,int], List[float]]:
    """
    Encodes one raw data row into categorical indices (ints) and numeric vector.
    Categorical indices are int tensors in [0, vocab_size-1] (0 is OOV).
    """
    cat_indices = {}
    for feat in vocab.keys():
        val = row.get(feat)
        if use_hashing:
            buckets = hash_buckets[feat]
            cat_indices[feat] = apply_hashing(val, buckets)
        else:
            mapping = vocab[feat]
            cat_indices[feat] = mapping.get(val, 0)  # 0 => OOV
    numeric_vector = [float(row.get(f, 0.0)) for f in numeric_feats]
    return cat_indices, numeric_vector

# -----------------------------
# Dataset and DataLoader
# -----------------------------
class TabularDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], labels: List[int],
                 vocab: Dict[str, Dict[Any,int]], numeric_feats: List[str],
                 use_hashing: bool=False, hash_buckets: Dict[str,int]=None):
        self.rows = rows
        self.labels = labels
        self.vocab = vocab
        self.numeric_feats = numeric_feats
        self.use_hashing = use_hashing
        self.hash_buckets = hash_buckets

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        cat_idx, num_vec = encode_row(row, self.vocab, self.numeric_feats, self.use_hashing, self.hash_buckets)
        # pack as tensors
        cat_tensor = {k: torch.tensor(v, dtype=torch.long) for k, v in cat_idx.items()}
        num_tensor = torch.tensor(num_vec, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return cat_tensor, num_tensor, label

def collate_batch(batch):
    # batch is list of (cat_tensor_dict, num_tensor, label)
    cat_keys = batch[0][0].keys()
    batch_cat = {}
    for k in cat_keys:
        batch_cat[k] = torch.stack([item[0][k] for item in batch]).unsqueeze(1)  # shape [B,1]
    batch_num = torch.stack([item[1] for item in batch])  # [B, num_feat]
    batch_labels = torch.stack([item[2] for item in batch]).unsqueeze(1)  # [B,1]
    # convert to shape [B] long tensors for embedding lookup
    for k in batch_cat:
        batch_cat[k] = batch_cat[k].view(len(batch),)  # [B]
    return batch_cat, batch_num, batch_labels

# -----------------------------
# Model
# -----------------------------
class TabularBinaryModel(nn.Module):
    def __init__(self,
                 vocab: Dict[str, Dict[Any,int]],
                 numeric_feat_count: int,
                 use_hashing: bool=False,
                 hash_buckets: Dict[str,int]=None,
                 embedding_max_dim: int=EMBEDDING_MAX_DIM,
                 embedding_power: float=EMBEDDING_DIM_HEURISTIC_POWER,
                 dropout_p: float=0.2):
        super().__init__()
        self.use_hashing = use_hashing
        self.emb_layers = nn.ModuleDict()
        self.feature_order = list(vocab.keys())
        self.embedding_output_dim = 0

        for feat in self.feature_order:
            if use_hashing:
                card = hash_buckets[feat]
                emb_dim = min(embedding_max_dim, max(4, int(card ** embedding_power)))
                self.emb_layers[feat] = nn.Embedding(card, emb_dim)
            else:
                card = len(vocab[feat])  # includes OOV
                emb_dim = min(embedding_max_dim, max(4, int(card ** embedding_power)))
                self.emb_layers[feat] = nn.Embedding(card, emb_dim)
            self.embedding_output_dim += self.emb_layers[feat].embedding_dim

        # small MLP
        mlp_in = self.embedding_output_dim + numeric_feat_count
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1)  # raw logit
        )

    def forward(self, cat_inputs: Dict[str, torch.LongTensor], numeric_inputs: torch.Tensor):
        emb_list = []
        for feat in self.feature_order:
            x = cat_inputs[feat].to(next(self.parameters()).device)
            emb = self.emb_layers[feat](x)
            emb_list.append(emb)
        emb_cat = torch.cat(emb_list, dim=1)  # [B, total_emb_dim]
        x = torch.cat([emb_cat, numeric_inputs.to(emb_cat.device)], dim=1)
        logit = self.mlp(x)
        return logit  # raw logit; use BCEWithLogitsLoss

# -----------------------------
# Synthetic data generator (for demo)
# -----------------------------
def generate_synthetic_data(n_rows: int = 200_000):
    rows = []
    labels = []
    # create biased distributions so top-K retention makes sense
    for i in range(n_rows):
        # heavy-tail user ids
        user = f"user_{random.randint(1, 400_000) if random.random() < 0.2 else random.randint(1, 50_000)}"
        merchant = f"merch_{random.randint(1, 120_000) if random.random() < 0.3 else random.randint(1, 30_000)}"
        mcc = f"mcc_{random.randint(1, 800)}"
        amt = max(0.0, random.gauss(50, 40))
        hour = random.randint(0, 23)
        # synthetic label: higher risk for certain merchants/users
        label = 1 if ("user_100" in user or "merch_500" in merchant and amt > 100) else (random.random() < 0.02)
        rows.append({"user_id": user, "merchant_id": merchant, "mcc_code": mcc,
                     "txn_amount": amt, "hour_of_day": hour})
        labels.append(int(label))
    return rows, labels

# -----------------------------
# Training loop + evaluation
# -----------------------------
def train_model(train_loader, val_loader, model, optimizer, epochs=EPOCHS, device=DEVICE):
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_cat, batch_num, batch_labels in train_loader:
            batch_cat = {k: v.to(device) for k,v in batch_cat.items()}
            batch_num = batch_num.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_cat, batch_num)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch_cat, batch_num, batch_labels in val_loader:
                batch_cat = {k: v.to(device) for k,v in batch_cat.items()}
                batch_num = batch_num.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_cat, batch_num)
                loss = criterion(logits, batch_labels)
                val_running += loss.item() * batch_labels.size(0)
        val_loss = val_running / len(val_loader.dataset)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save best weights
            best_state = model.state_dict()
    # return best weights (or final)
    model.load_state_dict(best_state)
    return model

# -----------------------------
# Save / Load helpers
# -----------------------------
def save_artifacts(model: nn.Module, vocab: Dict[str, Dict[Any,int]], model_path: str, vocab_path: str):
    torch.save(model.state_dict(), model_path)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print("Saved model & vocab.")

def load_artifacts(model_class, model_kwargs, model_path: str, vocab_path: str):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, vocab

# -----------------------------
# Inference pipeline for streaming data / daily batches
# -----------------------------
class InferenceTransform:
    def __init__(self, vocab: Dict[str, Dict[Any,int]], numeric_feats: List[str],
                 use_hashing: bool=False, hash_buckets: Dict[str,int]=None):
        self.vocab = vocab
        self.numeric_feats = numeric_feats
        self.use_hashing = use_hashing
        self.hash_buckets = hash_buckets

    def transform_one(self, raw_row: Dict[str, Any]):
        cat_idx, num_vec = encode_row(raw_row, self.vocab, self.numeric_feats,
                                      use_hashing=self.use_hashing, hash_buckets=self.hash_buckets)
        # convert to tensors shaped as single-batch
        cat_tensor = {k: torch.tensor([v], dtype=torch.long) for k, v in cat_idx.items()}
        num_tensor = torch.tensor([num_vec], dtype=torch.float32)
        return cat_tensor, num_tensor

    def predict_one(self, raw_row: Dict[str, Any], model: nn.Module, device=DEVICE):
        model.to(device)
        cat_t, num_t = self.transform_one(raw_row)
        cat_t = {k: v.to(device) for k, v in cat_t.items()}
        num_t = num_t.to(device)
        with torch.no_grad():
            logit = model(cat_t, num_t)
            prob = torch.sigmoid(logit).item()
        return prob

# -----------------------------
# Main: run demo training + save + example inference
# -----------------------------
def main_demo():
    print("Generating synthetic data...")
    rows, labels = generate_synthetic_data(120_000)
    # split
    split = int(0.8 * len(rows))
    train_rows, train_labels = rows[:split], labels[:split]
    val_rows, val_labels = rows[split:], labels[split:]

    print("Building vocab (frequency capping / OOV)...")
    vocab = build_vocab_from_data(train_rows, VOCAB_CONFIG)

    print("Creating datasets...")
    train_ds = TabularDataset(train_rows, train_labels, vocab, NUMERIC_FEATURES,
                              use_hashing=USE_HASHING, hash_buckets=HASH_BUCKETS if USE_HASHING else None)
    val_ds = TabularDataset(val_rows, val_labels, vocab, NUMERIC_FEATURES,
                            use_hashing=USE_HASHING, hash_buckets=HASH_BUCKETS if USE_HASHING else None)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    print("Instantiating model...")
    model = TabularBinaryModel(vocab=vocab, numeric_feat_count=len(NUMERIC_FEATURES),
                               use_hashing=USE_HASHING, hash_buckets=HASH_BUCKETS if USE_HASHING else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training ...")
    model = train_model(train_loader, val_loader, model, optimizer, epochs=EPOCHS, device=DEVICE)

    # Save artifacts
    save_artifacts(model, vocab, "tabular_model.pt", "vocab.pkl")

    # Example inference
    inf = InferenceTransform(vocab=vocab, numeric_feats=NUMERIC_FEATURES,
                             use_hashing=USE_HASHING, hash_buckets=HASH_BUCKETS if USE_HASHING else None)
    sample_row = {
        "user_id": "user_100", "merchant_id": "merch_500", "mcc_code": "mcc_10",
        "txn_amount": 320.5, "hour_of_day": 2
    }
    prob = inf.predict_one(sample_row, model, device=DEVICE)
    print(f"Predicted fraud probability for sample: {prob:.4f}")

if __name__ == "__main__":
    main_demo()
