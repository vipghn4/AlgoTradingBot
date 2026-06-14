import yfinance as yf

def get_market_risk_free_rate(ticker: str = '^TNX', fallback: float = 0.04) -> float:
    """
    Fetches the current risk-free rate from market data (default: 10Y Treasury).
    
    Args:
        ticker (str): Ticker symbol for the risk-free proxy.
        fallback (float): The rate to return if the fetch fails.
        
    Returns:
        float: The decimal representation of the yield (e.g., 0.045 for 4.5%).
    """
    try:
        tnx = yf.Ticker(ticker)
        history = tnx.history(period='1d')
        if not history.empty:
            return history['Close'].iloc[-1] / 100
    except Exception as e:
        print(f"Warning: Could not fetch risk-free rate ({e}). Using fallback.")
    
    return fallback