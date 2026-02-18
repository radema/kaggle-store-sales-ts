import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import torch.nn.functional as F

from src.models.gnn_v2.config import GNNConfig
from src.models.gnn_v2.graph import GraphFactory
from src.models.gnn_v2.loader import TemporalGraphDataset, make_gnn_v2_loader
from src.models.gnn_v2.model import SalesGNNv2
from src.utils.logging import get_logger

logger = get_logger("train_gnn_v2")

# Removed create_temporal_tensors to save memory - windowing is handled by dataset/loader.

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    count = 0
    start_time = time.time()
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        
        if torch.isnan(loss):
            logger.error("NaN loss detected!")
            return float('nan'), 0
            
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    duration = time.time() - start_time
    return total_loss / count if count > 0 else 0, duration

@torch.no_grad()
def validate(model, loader, criterion, device, max_batches=50):
    model.eval()
    total_loss = 0
    total_mse = 0
    count = 0
    
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch.y)
        total_loss += loss.item()
        
        # RMSLE calculation (Targets and inputs are already in log1p)
        mse = F.mse_loss(out, batch.y)
        total_mse += mse.item()
        
        count += 1
    
    avg_loss = total_loss / count if count > 0 else 0
    avg_rmsle = np.sqrt(total_mse / count) if count > 0 else 0
        
    return avg_loss, avg_rmsle

def main(args):
    config = GNNConfig()
    if args.dev:
        logger.info("Dev Mode: Overriding epochs and sampling")
        config.epochs = 2
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Load Data
    data_dir = Path("data/processed/gnn")
    raw_dir = Path("data/raw")
    
    logger.info("Loading features and labels...")
    features = torch.from_numpy(np.load(data_dir / "features.npy")).float()
    labels = torch.from_numpy(np.load(data_dir / "labels.npy")).float()
    
    # 2. Build Dataset
    # We NO LONGER expand to 4D tensors upfront to save memory.
    # The updated TemporalGraphDataset handles windowing on the fly.
    
    # Create families from train data
    train_df = pd.read_parquet("data/processed/train.parquet")
    families = sorted(train_df["family"].unique())
    
    factory = GraphFactory(raw_dir / "stores.csv", families)
    static_graph = factory.build_graph()
    
    dataset = TemporalGraphDataset(
        static_graph, features, labels, 
        window=config.window, 
        horizon=config.horizon
    )
    
    # 3. Split Indices (Temporal Split: last 16 days for val)
    total_steps = dataset.num_time_steps
    val_steps = 16 
    train_steps = total_steps - val_steps
    
    total_steps = dataset.num_time_steps
    val_steps = 16 
    train_steps = total_steps - val_steps
    
    if args.dev:
        train_steps_active = min(train_steps, 5)
        train_start_step = train_steps - train_steps_active
        val_steps_active = min(val_steps, 2)
        val_start_step = total_steps - val_steps_active
    else:
        train_start_step = 0
        train_steps_active = train_steps
        val_start_step = train_steps
        val_steps_active = val_steps

    def get_flat_indices(start_step, num_steps, num_nodes, shuffle_days=False):
        day_indices = list(range(start_step, start_step + num_steps))
        if shuffle_days:
            np.random.shuffle(day_indices)
        times = []
        nodes = []
        for t in day_indices:
            for n in range(num_nodes):
                times.append(t)
                nodes.append(n)
        return times, nodes

    train_times, train_nodes = get_flat_indices(train_start_step, train_steps_active, factory.num_nodes, shuffle_days=not args.dev)
    val_times, val_nodes = get_flat_indices(val_start_step, val_steps_active, factory.num_nodes, shuffle_days=False)
    
    # 5. Create Loaders
    batch_size = factory.num_nodes
    
    train_loader = make_gnn_v2_loader(
        dataset, train_times, train_nodes, 
        batch_size=batch_size,
        num_neighbors=config.neighbor_sizes,
        shuffle=False
    )
    
    val_loader = make_gnn_v2_loader(
        dataset, val_times, val_nodes,
        batch_size=batch_size,
        num_neighbors=config.neighbor_sizes,
        shuffle=False
    )
    
    # 6. Model
    model = SalesGNNv2(
        in_channels=features.size(-1),
        hidden_channels=config.hidden_dim,
        out_channels=config.horizon,
        num_layers=2,
        gat_heads=config.gat_heads,
        dropout=config.dropout
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.HuberLoss()
    
    # 7. Training Loop
    logger.info(f"Starting training for {config.epochs} epochs on {device}")
    best_val_loss = float('inf')
    early_stop_count = 0
    
    total_train_time = 0
    total_nodes_processed = 0
    peak_vram = 0
    
    for epoch in range(config.epochs):
        train_loss, duration = train_one_epoch(model, train_loader, optimizer, criterion, device)
        total_train_time += duration
        total_nodes_processed += len(train_times) # Total index pairs
        
        if "mps" in str(device):
            current_vram = torch.mps.current_allocated_memory() / (1024**3)
            peak_vram = max(peak_vram, current_vram)
        
        val_loss, val_rmsle = validate(model, val_loader, criterion, device, max_batches=500 if args.full_val else 50)
        
        logger.info(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val RMSLE: {val_rmsle:.6f} | Time: {duration:.2f}s")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            artifacts_dir = Path("artifacts/gnn_v2")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), artifacts_dir / "model_best.pt")
            logger.info(f"Saved new best model with Val Loss: {val_loss:.6f}")
            early_stop_count = 0
        else:
            early_stop_count += 1
            
        if early_stop_count >= config.early_stopping_patience:
            logger.info("Early stopping triggered.")
            break
            
    throughput = total_nodes_processed / total_train_time
    logger.info(f"Training finished. Best Val Loss: {best_val_loss:.6f}")
    logger.info(f"Throughput: {throughput:.2f} nodes/sec")
    if "mps" in str(device):
        logger.info(f"Peak VRAM: {peak_vram:.2f} GB")

    # 9. Submission Pass
    generate_submission(model, dataset, factory, device)

def generate_submission(model, dataset, factory, device):
    logger.info("Starting full inference for Kaggle submission...")
    
    # Load best model if exists
    best_model_path = Path("artifacts/gnn_v2/model_best.pt")
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"Loaded best model from {best_model_path}")
    
    model.eval()
    
    # The submission period starts right after the last day in our features (if features are historical only)
    # OR we use the very last available window to predict the next 16 days.
    # In temporal competition data, features usually include the test period.
    # We use the window ending at the last training day.
    
    # Total days in features: features.size(0)
    # If the last day of features is the day before test starts:
    last_window_start = dataset.features.size(0) - dataset.window
    
    times_inf = [last_window_start] * dataset.num_nodes
    nodes_inf = list(range(dataset.num_nodes))
    
    # Use batch_size = num_nodes for one single pass
    inf_loader = make_gnn_v2_loader(dataset, times_inf, nodes_inf, batch_size=dataset.num_nodes, shuffle=False)
    
    batch = next(iter(inf_loader)).to(device)
    with torch.no_grad():
        out = model(batch) # (Num_Nodes, Horizon=16)
        
    # Invert log transform: log1p -> raw sales
    preds_raw = torch.expm1(out).cpu().numpy()
    preds_raw = np.maximum(preds_raw, 0) # Clamp negatives
    
    # MAP TO KAGGLE ORDER
    # Kaggle expects: Date 1 (All Nodes), Date 2 (All Nodes)...
    # Our out is: Node 1 (All Dates), Node 2 (All Dates)...
    # Thus, we need to transpose to (Horizon, Nodes) before flattening.
    preds_kag = preds_raw.T.flatten() # (16, 1782) -> (28512,)
    
    # Load test.csv to get the IDs
    test_df = pd.read_csv("data/raw/test.csv")
    if len(preds_kag) != len(test_df):
        logger.error(f"Prediction mismatch! Model: {len(preds_kag)}, Test.csv: {len(test_df)}")
        # If mismatch, might be because features.npy doesn't end exactly at T_train.
        # Let's adjust if needed.
        return

    submission = pd.DataFrame({
        "id": test_df["id"],
        "sales": preds_kag
    })
    
    output_path = Path("artifacts/gnn_v2/submission.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true", help="Run in dev mode (2 epochs, small data)")
    parser.add_argument("--full-val", action="store_true", help="Run full validation instead of a subset")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for loader")
    parser.add_argument("--predict-only", action="store_true", help="Skip training and just generate submission.csv")
    args = parser.parse_args()
    
    if args.predict_only:
        # Minimal setup for prediction
        config = GNNConfig()
        data_dir = Path("data/processed/gnn")
        features = torch.from_numpy(np.load(data_dir / "features.npy")).float()
        labels = torch.from_numpy(np.load(data_dir / "labels.npy")).float()
        factory = GraphFactory(Path("data/raw/stores.csv"), sorted(pd.read_parquet("data/processed/train.parquet")["family"].unique()))
        static_graph = factory.build_graph()
        dataset = TemporalGraphDataset(static_graph, features, labels, config.window, config.horizon)
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model = SalesGNNv2(
            in_channels=features.size(-1),
            hidden_channels=config.hidden_dim,
            out_channels=config.horizon,
            num_layers=2,
            gat_heads=config.gat_heads,
            dropout=config.dropout
        ).to(device)
        generate_submission(model, dataset, factory, device)
    else:
        main(args)
