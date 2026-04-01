import pandas as pd
DATA_PATH = r'C:\Users\Hemant\OneDrive\Documents\car_prices.csv'
try:
    df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
    print("Columns:", df.columns.tolist())
    print("Types:\n", df.dtypes)
    # Check for non-numeric values in numeric columns
    for col in ['year', 'condition', 'odometer', 'mmr', 'sellingprice']:
        print(f"Checking column: {col}")
        # Try to convert to numeric and find where it fails
        bad_idx = pd.to_numeric(df[col], errors='coerce').isna()
        if bad_idx.any():
            print(f"Found non-numeric values in {col}:")
            print(df.loc[bad_idx, col].unique()[:10])
except Exception as e:
    print(f"Error: {e}")
