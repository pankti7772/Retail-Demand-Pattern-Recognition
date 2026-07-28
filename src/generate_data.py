"""
generate_data.py
-----------------
Creates a synthetic, M5-style retail dataset so the MVP pipeline can run
end-to-end without needing to download Kaggle datasets (M5 / Rossmann /
Walmart). Swap this out later for the real dataset -- the pipeline only
needs a dataframe with these columns:

    date, store_id, product_id, category, region, price, base_price,
    promo_flag, is_holiday, is_festival, units_sold

Run:
    python src/generate_data.py
Produces:
    data/raw_sales.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_DAYS = 730          # 2 years of daily data
N_PRODUCTS = 60
N_STORES = 4
START_DATE = "2024-01-01"

REGIONS = ["North", "South", "East", "West"]
STORE_REGION = {f"STORE_{i+1}": REGIONS[i % len(REGIONS)] for i in range(N_STORES)}

# Each product belongs to a "true" behaviour type -- this is the pattern
# the clustering + forecasting steps are meant to (re)discover.
CATEGORY_PROFILE = {
    "Fast_Moving_Essentials": dict(base_demand=(80, 150), trend=0.00, season_amp=0.05, promo_lift=0.10, price_elasticity=-0.3),
    "Seasonal_Products":      dict(base_demand=(20, 60),  trend=0.02, season_amp=0.65, promo_lift=0.15, price_elasticity=-0.5),
    "Luxury_Products":        dict(base_demand=(5, 20),   trend=0.01, season_amp=0.20, promo_lift=0.05, price_elasticity=-1.2),
    "Highly_Promotional":     dict(base_demand=(30, 70),  trend=0.00, season_amp=0.15, promo_lift=0.55, price_elasticity=-0.8),
}
CATEGORIES = list(CATEGORY_PROFILE.keys())


def build_calendar(start_date: str, n_days: int) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    cal = pd.DataFrame({"date": dates})
    cal["day_of_week"] = cal["date"].dt.dayofweek  # 0=Mon
    cal["is_weekend"] = (cal["day_of_week"] >= 5).astype(int)
    cal["month"] = cal["date"].dt.month
    cal["quarter"] = cal["date"].dt.quarter
    cal["week_number"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["day_of_year"] = cal["date"].dt.dayofyear

    # A handful of fixed "holidays" per year (day-of-year based, so it repeats yearly)
    holiday_doy = {1, 50, 75, 110, 190, 226, 280, 300, 359}
    cal["is_holiday"] = cal["day_of_year"].apply(lambda d: int((d % 365) in holiday_doy))

    # Festival season seasonality (e.g. Oct-Dec "festive quarter" boost)
    cal["is_festival"] = cal["month"].isin([10, 11, 12]).astype(int)
    return cal


def simulate_product_catalog(n_products: int) -> pd.DataFrame:
    rows = []
    for i in range(n_products):
        category = CATEGORIES[i % len(CATEGORIES)]
        profile = CATEGORY_PROFILE[category]
        base_demand = RNG.uniform(*profile["base_demand"])
        base_price = round(RNG.uniform(50, 500), 2)
        rows.append(dict(
            product_id=f"PROD_{i+1:03d}",
            category=category,
            base_demand=base_demand,
            base_price=base_price,
            trend=profile["trend"] * RNG.uniform(0.7, 1.3),
            season_amp=profile["season_amp"] * RNG.uniform(0.8, 1.2),
            promo_lift=profile["promo_lift"] * RNG.uniform(0.8, 1.2),
            price_elasticity=profile["price_elasticity"] * RNG.uniform(0.8, 1.2),
            # random seasonal phase so not every product peaks the same week
            phase=RNG.uniform(0, 2 * np.pi),
        ))
    return pd.DataFrame(rows)


def simulate_sales(calendar: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    stores = list(STORE_REGION.keys())
    n_days = len(calendar)
    all_rows = []

    for _, prod in products.iterrows():
        # Promotions run in irregular bursts (~12% of days), more likely for promo-driven products
        promo_prob = 0.28 if prod["category"] == "Highly_Promotional" else 0.10
        promo_flag = (RNG.random(n_days) < promo_prob).astype(int)

        # Price fluctuates a little day to day, with a discount during promos
        price_noise = RNG.normal(0, 0.02, n_days)
        promo_discount = np.where(promo_flag == 1, RNG.uniform(0.10, 0.30, n_days), 0.0)
        price = prod["base_price"] * (1 + price_noise) * (1 - promo_discount)

        day_idx = np.arange(n_days)
        seasonal = 1 + prod["season_amp"] * np.sin(2 * np.pi * day_idx / 365 + prod["phase"])
        trend = 1 + prod["trend"] * (day_idx / 365)
        weekend_lift = 1 + 0.20 * calendar["is_weekend"].values
        holiday_lift = 1 + 0.35 * calendar["is_holiday"].values
        festival_lift = 1 + (0.5 if prod["category"] == "Seasonal_Products" else 0.15) * calendar["is_festival"].values
        promo_lift = 1 + prod["promo_lift"] * promo_flag
        price_effect = (price / prod["base_price"]) ** prod["price_elasticity"]

        mean_demand = (
            prod["base_demand"] * seasonal * trend * weekend_lift
            * holiday_lift * festival_lift * promo_lift * price_effect
        )
        mean_demand = np.clip(mean_demand, 1, None)

        # Negative-binomial-ish noise via Poisson w/ gamma-mixed rate (overdispersion)
        rate = RNG.gamma(shape=8.0, scale=mean_demand / 8.0)
        units = RNG.poisson(rate)

        # Inject anomalies: ~1.5% of days get a random spike or a stock-out style crash
        anomaly_mask = RNG.random(n_days) < 0.015
        spike = RNG.random(n_days) < 0.5
        units = units.astype(float)
        units[anomaly_mask & spike] *= RNG.uniform(3, 6)
        units[anomaly_mask & ~spike] *= RNG.uniform(0.0, 0.15)
        units = np.round(units).astype(int)

        # A few missing values / duplicate-style dirty rows for the cleaning step to handle
        missing_mask = RNG.random(n_days) < 0.01

        for store in stores:
            store_mult = RNG.uniform(0.7, 1.3)  # store-level scale
            store_units = np.round(units * store_mult).astype(float)
            store_units[missing_mask] = np.nan

            df = pd.DataFrame({
                "date": calendar["date"].values,
                "store_id": store,
                "region": STORE_REGION[store],
                "product_id": prod["product_id"],
                "category": prod["category"],
                "price": np.round(price, 2),
                "base_price": prod["base_price"],
                "promo_flag": promo_flag,
                "is_holiday": calendar["is_holiday"].values,
                "is_festival": calendar["is_festival"].values,
                "units_sold": store_units,
            })
            all_rows.append(df)

    sales = pd.concat(all_rows, ignore_index=True)

    # Inject a small number of exact-duplicate rows to exercise dedup logic
    dupes = sales.sample(frac=0.002, random_state=1)
    sales = pd.concat([sales, dupes], ignore_index=True)

    return sales


def main():
    calendar = build_calendar(START_DATE, N_DAYS)
    products = simulate_product_catalog(N_PRODUCTS)
    sales = simulate_sales(calendar, products)

    calendar.to_csv("data/calendar.csv", index=False)
    products.drop(columns=["phase"]).to_csv("data/products.csv", index=False)
    sales.to_csv("data/raw_sales.csv", index=False)

    print(f"Generated {len(sales):,} rows -> data/raw_sales.csv")
    print(f"Products: {N_PRODUCTS} | Stores: {N_STORES} | Days: {N_DAYS}")


if __name__ == "__main__":
    main()
