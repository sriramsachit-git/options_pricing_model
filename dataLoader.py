import pandas as pd
import yfinance as yf
from datetime import datetime

def fetch_options_data(ticker, expiration_date):
    """
    Fetch options data for a given ticker and expiration date.

    Parameters:
    ticker (str): Ticker symbol to fetch options data for.
    expiration_date (str): Expiration date of options in 'YYYY-MM-DD' format.

    Returns:
    pandas.DataFrame: DataFrame with options data including strike prices and other details,
                      or an empty DataFrame if the expiration date is unavailable.
    """
    stock = yf.Ticker(ticker)

    # Get available expiration dates for the ticker
    available_expirations = stock.options
    if expiration_date not in available_expirations:
        print(f"Error: Expiration `{expiration_date}` is not available for {ticker}.")
        print(f"Available expirations are: {available_expirations}")
        return pd.DataFrame()  # Return an empty DataFrame

    # Fetch options chain for the valid expiration date
    try:
        options_chain = stock.option_chain(expiration_date)
        calls = options_chain.calls
        puts = options_chain.puts

        # Combine calls and puts into a single DataFrame with a type column
        calls["type"] = "call"
        puts["type"] = "put"
        options_data = pd.concat([calls, puts], ignore_index=True)

        return options_data
    except Exception as e:
        print(f"Error fetching options data for {ticker} on {expiration_date}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Testing the function
if __name__ == "__main__":
    ticker = "AAPL"
    expiration_date = "2025-01-17"  # Change this to a valid expiration date if needed
    print(f"Testing fetch_options_data for {ticker} with expiration date {expiration_date}...")
    
    options_data = fetch_options_data(ticker, expiration_date)
    if options_data.empty:
        print("No data available for the specified expiration date.")
    else:
        print("Options data retrieved successfully:")
        print(options_data.head())
