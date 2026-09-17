import pandas as pd
import numpy as np
import os

def load_and_process_data(data_dir='./../data'):
    """
    Load and process CSV files containing stock data.
    
    Args:
        data_dir (str): Directory containing CSV files
        
    Returns:
        dict: Dictionary mapping stock names to DataFrames with processed features
    """
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), data_dir))

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Dictionary to store DataFrames
    data_dict = {}
    
    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    for filename in csv_files:
        # Extract stock name from filename
        stock_name = filename.replace('.csv', '')
        
        # Load CSV file
        df = pd.read_csv(os.path.join(data_dir, filename), 
                        parse_dates=['Date'], 
                        index_col='Date')
        
        # Select required columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Convert Volume to numeric by removing commas and handle missing values
        if df['Volume'].dtype == object or isinstance(df['Volume'].iloc[0], str):
            df['Volume'] = df['Volume'].astype(str).str.replace(',', '', regex=False)
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

        df['Volume'] = df['Volume'].fillna(0).astype(float)

        # Calculate returns
        df['returns'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Calculate SMA indicators using pandas rolling mean
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate trend strength
        df['trend_strength'] = (df['SMA50'] - df['SMA200']) / df['SMA200']
        
        # Calculate RSI using pandas (14-day period)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Calculate momentum
        df['momentum'] = df['returns'].rolling(window=10).sum()
        
        # Calculate Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (std * 2)
        df['BB_lower'] = df['BB_middle'] - (std * 2)
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Add stock name column
        df['stock_name'] = stock_name
        
        # Store in dictionary
        data_dict[stock_name] = df
    
    return data_dict

def get_combined_data(data_dict, start_date='2019-01-01', end_date='2021-12-31'):
    """
    Combine DataFrames from multiple stocks and filter by date range.
    
    Args:
        data_dict (dict): Dictionary mapping stock names to DataFrames
        start_date (str): Start date for filtering (YYYY-MM-DD)
        end_date (str): End date for filtering (YYYY-MM-DD)
        
    Returns:
        DataFrame: Combined DataFrame with data from all stocks
    """
    filtered_dfs = []
    
    for stock_name, df in data_dict.items():
        # Filter by date range
        filtered_df = df.loc[start_date:end_date].copy()
        # Add to list if not empty
        if not filtered_df.empty:
            filtered_dfs.append(filtered_df)
    
    # Concatenate all DataFrames
    if filtered_dfs:
        combined_df = pd.concat(filtered_dfs)
        # Sort by date
        combined_df.sort_index(inplace=True)
        return combined_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    data_dict = load_and_process_data(data_dir='../data')
    combined_df = get_combined_data(data_dict)
    print(f"Loaded data for {len(data_dict)} stocks")
    print(f"Combined DataFrame shape: {combined_df.shape}")
