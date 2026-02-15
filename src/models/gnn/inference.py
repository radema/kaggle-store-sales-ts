import torch
import pandas as pd
import numpy as np
from typing import Optional
from src.models.gnn.model import SalesGNN
from src.utils.logging import get_logger

logger = get_logger("gnn_inference")

class GNNInference:
    """
    Handles recursive forecasting and submission generation for SalesGNN.
    Ensures alignment with lexical sort order of (store_nbr, family).
    """

    def __init__(
        self, 
        model: SalesGNN, 
        device: torch.device,
        scaler: Optional[object] = None
    ):
        self.model = model.eval().to(device)
        self.device = device
        self.scaler = scaler

    def predict_recursive(
        self, 
        x_enc: torch.Tensor, 
        is_open: torch.Tensor,
        total_steps: int
    ) -> torch.Tensor:
        """
        Predict total_steps recursively if total_steps > model.horizon.
        x_enc: (1, N_series, Window, Features)
        is_open: (1, N_series, total_steps)
        """
        horizon = self.model.horizon
        all_preds = []
        
        # Current input buffer
        curr_x = x_enc.to(self.device)
        steps_done = 0
        
        with torch.no_grad():
            while steps_done < total_steps:
                # Predict next chunk
                # Mask needs to be for the current chunk
                mask_chunk = is_open[:, :, steps_done:steps_done+horizon].to(self.device)
                
                # Forward pass: (1, N_series, Horizon)
                y_pred = self.model(curr_x, is_open=mask_chunk)
                
                all_preds.append(y_pred.cpu())
                steps_done += horizon
                
                if steps_done < total_steps:
                    # Update curr_x for next recursive step if needed
                    # Note: STGNN usually requires exogenous features for tomorrow.
                    # If features include lags of sales, we would update them here.
                    # For a simple TCN, it might just need the historical window.
                    # Adjusting curr_x by shifting window and populating with y_pred
                    # This depends on feature engineering logic.
                    logger.warning("Recursive update of features not fully implemented - requires feature dependency map.")
        
        return torch.cat(all_preds, dim=-1)[:, :, :total_steps]

    def generate_submission(
        self,
        y_pred: torch.Tensor,
        test_df: pd.DataFrame,
        stores_sorted: pd.DataFrame,
        families_sorted: list
    ) -> pd.DataFrame:
        """
        Maps the (Batch, N_series, Horizon) output back to the test_df format.
        y_pred: (Batch, N_series, Horizon) -> assuming Batch=1 for inference
        """
        # 1. Flatten predictions
        # Lexical order: Store 1 (F1, F2...), Store 2 (F1, F2...)
        num_stores = len(stores_sorted)
        num_families = len(families_sorted)
        
        # Reshape to (Horizon, num_stores * num_families)
        y_flat = y_pred[0].t().numpy() # (Horizon, N_series)
        
        # 2. Inverse Transform (log1p -> sales)
        y_sales = np.expm1(y_flat)
        
        # 3. Create mapping for test_df
        # test_df should be sorted lexically to match
        test_df = test_df.sort_values(["date", "store_nbr", "family"])
        
        # Verify length
        if len(test_df) != y_sales.size:
            logger.error(f"Shape mismatch! Test df: {len(test_df)}, Preds: {y_sales.size}")
            # Attempt to align by index if possible, otherwise raise
        
        test_df["sales"] = y_sales.flatten()
        return test_df.sort_values("id") # Return to original order for submission

def align_lexical_order(df: pd.DataFrame, stores: list, families: list):
    """
    Ensures the dataframe rows are ordered consistently with our Node indexing.
    Order: Store 1 [Fam 1, Fam 2...], Store 2 [Fam 1, Fam 2...]
    """
    df["store_nbr"] = pd.Categorical(df["store_nbr"], categories=stores, ordered=True)
    df["family"] = pd.Categorical(df["family"], categories=families, ordered=True)
    return df.sort_values(["store_nbr", "family"])
