import numpy as np
import pandas as pd


class SlidingWindowTS:
    """
    Sliding Window Time Series Splitter.
    Moves backwards from the most recent date.

    Parameters:
    -----------
    train_days : int
        Number of days for training in each fold.
    val_days : int
        Number of days for validation in each fold.
    n_folds : int
        Number of folds to generate.
    step_days : int, optional
        Number of days to shift the window in each step.
        Defaults to val_days (contiguous folds).
    """

    def __init__(self, train_days, val_days, n_folds=5, step_days=None):
        self.train_days = train_days
        self.val_days = val_days
        self.n_folds = n_folds
        self.step_days = step_days or val_days

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and validation set.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            if "date" in X.columns:
                dates = pd.to_datetime(X["date"])
            else:
                raise ValueError("X must have a DatetimeIndex or a 'date' column.")
        else:
            dates = X.index

        unique_dates = np.sort(dates.unique())
        n_unique = len(unique_dates)

        # We start from the end of the dataset
        for i in range(self.n_folds):
            # Calculate indices based on unique dates
            # Fold i: end of val is n_unique - (i * step_days)
            end_val_ptr = n_unique - (i * self.step_days)
            start_val_ptr = end_val_ptr - self.val_days
            start_train_ptr = start_val_ptr - self.train_days

            if start_train_ptr < 0:
                # Stop if we don't have enough data for the full train/val window
                break

            val_dates = unique_dates[start_val_ptr:end_val_ptr]
            train_dates = unique_dates[start_train_ptr:start_val_ptr]

            # Find row indices for these dates
            # Using isin on the original dates series/index
            val_mask = np.isin(dates, val_dates)
            train_mask = np.isin(dates, train_dates)

            yield np.where(train_mask)[0], np.where(val_mask)[0]
