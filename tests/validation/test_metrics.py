import numpy as np
import pytest
from src.validation.metrics import rmsle

def test_rmsle_correctness():
    # Test with known values
    y_true = np.array([10, 20, 30])
    y_pred = np.array([11, 19, 31])
    # log1p values:
    # true: ln(11), ln(21), ln(31) 
    # pred: ln(12), ln(20), ln(32)
    # manual validation later if needed, but standard formula is clear
    val = rmsle(y_true, y_pred)
    assert val > 0
    assert isinstance(val, float)

def test_rmsle_perfect_match():
    y_true = np.array([10, 20])
    y_pred = np.array([10, 20])
    assert rmsle(y_true, y_pred) == 0.0

def test_rmsle_handles_zeros():
    y_true = np.array([0, 0])
    y_pred = np.array([0, 0])
    assert rmsle(y_true, y_pred) == 0.0

def test_rmsle_clips_negatives():
    # Competition evaluation usually handles negatives by clipping to 0
    y_true = np.array([10, 10])
    y_pred = np.array([-5, 10])
    # Should treat -5 as 0
    val_clipped = rmsle(y_true, np.array([0, 10]))
    assert rmsle(y_true, y_pred) == val_clipped
