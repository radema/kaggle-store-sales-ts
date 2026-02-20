import pandas as pd
import numpy as np
import sys
import yaml
from pathlib import Path
from src.utils.logging import get_logger


def log1p(x):
    """Natural log scaling helper."""
    return np.log1p(x)


def expm1(x):
    """Inverse natural log scaling helper."""
    return np.expm1(x)


def load_data(logger, raw_dir="data/raw"):
    """Loads all raw datasets and returns a dictionary of DataFrames."""
    raw_path = Path(raw_dir)
    datasets = {}
    files = {
        "train": "train.csv",
        "test": "test.csv",
        "stores": "stores.csv",
        "oil": "oil.csv",
        "holidays": "holidays_events.csv",
        "transactions": "transactions.csv",
    }

    for name, filename in files.items():
        path = raw_path / filename
        if not path.exists():
            logger.error(f"File not found: {path}")
            sys.exit(1)

        logger.info(f"Loading {name} from {path}...")
        datasets[name] = pd.read_csv(path)

        # Immediate date conversion for relevant tables
        if "date" in datasets[name].columns:
            datasets[name]["date"] = pd.to_datetime(datasets[name]["date"])

    return datasets


def process_unification(logger, datasets):
    """Concatenates train/test and performs initial joins."""
    train = datasets["train"].copy()
    test = datasets["test"].copy()

    train["_set"] = "train"
    test["_set"] = "test"
    test["sales"] = 0.0  # Dummy for concat

    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    logger.info(f"Unified dataset shape: {df.shape}")

    # Store merge
    row_count_before = len(df)
    df = df.merge(datasets["stores"], on="store_nbr", how="left")
    logger.info(f"Merged stores. Row count: {row_count_before} -> {len(df)}")

    # Oil merge
    oil = datasets["oil"].copy()
    # Create complete date range to interpolate
    full_dates = pd.date_range(oil["date"].min(), oil["date"].max(), freq="D")
    oil = (
        oil.set_index("date")
        .reindex(full_dates)
        .reset_index()
        .rename(columns={"index": "date"})
    )
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(method="linear").bfill()

    row_count_before = len(df)
    df = df.merge(oil, on="date", how="left")
    logger.info(f"Merged oil. Row count: {row_count_before} -> {len(df)}")

    # Transactions merge
    row_count_before = len(df)
    df = df.merge(datasets["transactions"], on=["date", "store_nbr"], how="left")
    df["transactions"] = df["transactions"].fillna(0)
    logger.info(f"Merged transactions. Row count: {row_count_before} -> {len(df)}")

    return df


def process_holidays(logger, df, holidays_df):
    """Processes holidays and merges binary flags."""
    # Filter transferred holidays (they are shifted to another date)
    holidays = holidays_df[~holidays_df["transferred"]].copy()

    # National Holidays
    nat = holidays[holidays["locale"] == "National"].drop_duplicates("date").copy()
    nat["is_nat_holiday"] = 1

    # Regional Holidays (by State)
    reg = (
        holidays[holidays["locale"] == "Regional"]
        .drop_duplicates(["date", "locale_name"])
        .copy()
    )
    reg = reg.rename(columns={"locale_name": "state"})
    reg["is_reg_holiday"] = 1

    # Local Holidays (by City)
    loc = (
        holidays[holidays["locale"] == "Local"]
        .drop_duplicates(["date", "locale_name"])
        .copy()
    )
    loc = loc.rename(columns={"locale_name": "city"})
    loc["is_loc_holiday"] = 1

    row_count_before = len(df)

    # Sequential merges
    df = df.merge(nat[["date", "is_nat_holiday"]], on="date", how="left")
    df = df.merge(
        reg[["date", "state", "is_reg_holiday"]], on=["date", "state"], how="left"
    )
    df = df.merge(
        loc[["date", "city", "is_loc_holiday"]], on=["date", "city"], how="left"
    )

    # Specific Holiday Heuristics
    df["is_christmas_eve"] = (
        (df["date"].dt.month == 12) & (df["date"].dt.day == 24)
    ).astype(int)
    df["is_valentines_day"] = (
        (df["date"].dt.month == 2) & (df["date"].dt.day == 14)
    ).astype(int)
    df["is_mothers_day"] = (
        (df["date"].dt.month == 5)
        & (df["date"].dt.dayofweek == 6)
        & (df["date"].dt.day >= 8)
        & (df["date"].dt.day <= 14)
    ).astype(int)
    df["is_earthquake_period"] = (
        (df["date"] >= "2016-04-16") & (df["date"] <= "2016-05-14")
    ).astype(int)
    df["is_black_friday"] = (
        (df["date"].dt.month == 11)
        & (df["date"].dt.dayofweek == 4)
        & (df["date"].dt.day >= 24)
    ).astype(int)

    # Fill NAs with 0
    new_holiday_cols = [
        "is_christmas_eve",
        "is_valentines_day",
        "is_mothers_day",
        "is_earthquake_period",
        "is_black_friday",
    ]
    holiday_cols = [
        "is_nat_holiday",
        "is_reg_holiday",
        "is_loc_holiday",
    ] + new_holiday_cols
    df[holiday_cols] = df[holiday_cols].fillna(0).astype(int)

    logger.info(f"Merged holidays. Row count: {row_count_before} -> {len(df)}")
    return df


def process_domain_features(logger, df):
    """Calculates calendar features, rolling stats, proxies, and robust lags."""
    logger.info("Starting domain feature engineering...")

    # 1. Target Scaling
    df["log1p_sales"] = log1p(df["sales"])

    # 2. Calendar Features
    df["year"] = df["date"].dt.year.astype(np.int16)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["day"] = df["date"].dt.day.astype(np.int8)
    df["dayofweek"] = df["date"].dt.dayofweek.astype(np.int8)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
    df["is_wage_day"] = ((df["day"] == 15) | (df["date"].dt.is_month_end)).astype(
        np.int8
    )
    df["is_wage_day_weekend"] = (df["is_wage_day"] & df["is_weekend"]).astype(np.int8)

    # 2b. Region-based seasonality (Back to School)
    coast_provinces = [
        "Guayas",
        "Manabi",
        "Esmeraldas",
        "Los Rios",
        "El Oro",
        "Santa Elena",
    ]
    is_coast = df["state"].isin(coast_provinces)
    month = df["date"].dt.month
    df["is_back_to_school"] = (
        (is_coast & month.isin([4, 5])) | (~is_coast & month.isin([8, 9]))
    ).astype(np.int8)

    df["is_year_start"] = (df["date"].dt.is_year_start).astype(np.int8)
    df["is_year_end"] = (df["date"].dt.is_year_end).astype(np.int8)

    # Cyclic seasonality
    day_of_year = df["date"].dt.dayofyear
    df["sin_day_year"] = np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32)
    df["cos_day_year"] = np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32)

    # Cyclic seasonality for week and month
    day_of_month = df["date"].dt.day
    days_in_month = df["date"].dt.days_in_month

    df["sin_day_month"] = np.sin(2 * np.pi * day_of_month / days_in_month).astype(
        np.float32
    )
    df["cos_day_month"] = np.cos(2 * np.pi * day_of_month / days_in_month).astype(
        np.float32
    )

    df["sin_day_week"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)
    df["cos_day_week"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0).astype(np.float32)

    # 3. Rolling 30-day Mean (excluding current row via shift)
    # We sort by date to ensure rolling works correctly
    df = df.sort_values(["store_nbr", "family", "date"])

    df["rolling_30_sales"] = (
        df.groupby(["store_nbr", "family"])["log1p_sales"]
        .transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
        .fillna(0)
    )

    # 4. is_closed Flag
    # Re-define to capture true store closure
    # Jan 1st is structurally closed (excluding specific exceptions if we know them, but generally 0)
    # Also closed if the store had 0 transactions AND 0 sales on that date.
    # Note: transactions are already merged and NA filled with 0.

    # Store-level daily sum of sales
    store_daily_sales = df.groupby(["store_nbr", "date"])["sales"].transform("sum")

    df["is_closed"] = (
        ((df["date"].dt.month == 1) & (df["date"].dt.day == 1))
        | ((store_daily_sales == 0) & (df["transactions"] == 0))
    ).astype(np.int8)

    logger.info(
        f"Redefined is_closed explicitly. {df['is_closed'].sum()} rows marked as closed."
    )

    # 5. Target Encodings (Rolling 30-day Mean & 1-Year Lag)
    levels = ["cluster", "city", "type"]
    for level in levels:
        logger.info(f"Computing target encodings for {level}...")

        # 30-day Rolling Average
        proxy_name = f"{level}_sales_proxy"
        # We use existing cluster_sales_proxy name for compatibility with lag section
        if level == "cluster":
            proxy_name = "cluster_sales_proxy"

        level_proxy = (
            df.groupby(["date", level, "family"])["log1p_sales"]
            .mean()
            .reset_index()
            .rename(columns={"log1p_sales": proxy_name})
        )
        # Shift to avoid leakage! Merge back and then shift per group
        df = df.merge(level_proxy, on=["date", level, "family"], how="left")

        # We need the 30-day rolling mean of this proxy, shifted by 1 day
        df[f"rolling_30_{level}_sales"] = (
            df.groupby(["store_nbr", "family"])[proxy_name]
            .transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
            .fillna(0)
        )

        # 1-Year (364 days) Lag of the proxy to capture strong annual seasonality
        df[f"lag_364_{level}_sales"] = (
            df.groupby(["store_nbr", "family"])[proxy_name].shift(364).fillna(0)
        )

        # Clean up the intermediate proxy column (except cluster which is used in lags)
        if level != "cluster":
            df = df.drop(columns=[proxy_name])

    # 5b. Exact Annual Lag with cluster fallback (Coalesce)
    logger.info("Computing lag_364_sales (exact with cluster fallback)...")
    df["lag_364_sales"] = (
        df.groupby(["store_nbr", "family"])["log1p_sales"]
        .shift(364)
        .fillna(df["lag_364_cluster_sales"])
        .fillna(0)
    )

    # 6. Robust Lags [7, 14, 21, 28]
    lags = [7, 14, 21, 28]
    for lag in lags:
        logger.info(f"Calculating lag {lag}...")
        col_name = f"lag_{lag}"
        proxy_col_name = f"lag_{lag}_proxy"

        # Actual lag
        df[col_name] = df.groupby(["store_nbr", "family"])["log1p_sales"].shift(lag)

        # Proxy lag (using the same shift on the cluster-family-date average)
        df[proxy_col_name] = df.groupby(["store_nbr", "family"])[
            "cluster_sales_proxy"
        ].shift(lag)

        # Impute missing actual lags with proxy lags
        df[col_name] = df[col_name].fillna(df[proxy_col_name])

        # Cleanup proxy column
        df = df.drop(columns=[proxy_col_name])

    # Drop intermediate proxy column
    df = df.drop(columns=["cluster_sales_proxy"])

    return df


def save_feature_catalog(logger, df, path):
    """Generates a YAML feature catalog from the processed training dataframe."""
    logger.info(f"Generating feature catalog at {path}...")

    catalog = {
        "metadata": ["id", "date"],
        "target": ["sales", "log1p_sales"],
        "categorical_features": [],
        "numerical_features": [],
        "all_features": [],
    }

    # Exclude meta and target for feature lists
    exclude = catalog["metadata"] + catalog["target"]

    for col in df.columns:
        if col in exclude:
            continue

        catalog["all_features"].append(col)
        if df[col].dtype in ["object", "category"]:
            catalog["categorical_features"].append(col)
        else:
            catalog["numerical_features"].append(col)

    with open(path, "w") as f:
        yaml.dump(catalog, f, default_flow_style=False, sort_keys=False)

    logger.info("Feature catalog saved.")


def main():
    logger = get_logger("preprocess")
    logger.info("Starting preprocessing pipeline...")

    # Phase 1: Foundation
    datasets = load_data(logger)

    # Phase 2: Unification & Joins
    df = process_unification(logger, datasets)
    df = process_holidays(logger, df, datasets["holidays"])

    # Phase 3: Domain Features
    df = process_domain_features(logger, df)

    # Phase 4: Finalization
    # Uniqueness Check
    max_count = df.groupby(["date", "store_nbr", "family"]).size().max()
    if max_count > 1:
        logger.error(
            f"Integrity Error: Found {max_count} duplicates for (date, store_nbr, family)"
        )
        sys.exit(1)

    # Saving
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_out = df[df["_set"] == "train"].drop(columns=["_set"])
    test_out = df[df["_set"] == "test"].drop(columns=["_set", "sales", "log1p_sales"])

    train_out.to_parquet(data_dir / "train.parquet", index=False)
    test_out.to_parquet(data_dir / "test.parquet", index=False)

    # Building Feature Catalog from train
    save_feature_catalog(logger, train_out, data_dir / "feature_catalog.yaml")

    logger.info(f"Final dataset shapes: Train {train_out.shape}, Test {test_out.shape}")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
