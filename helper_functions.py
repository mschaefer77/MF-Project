import pandas as pd 
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def GeomSummary(returns: pd.Series, risk_free_rate: float = 0.0, confidence: float = 0.05, periods_per_year: int = 252) -> pd.Series:
    """
    Summary of return metrics using geometric mean in ratio calculations.
    
    Parameters:
        returns: pd.Series of daily arithmetic returns
        risk_free_rate: daily risk-free rate (default 0)
        confidence: for Value at Risk (VaR), default is 5%
        periods_per_year: default 252 (daily returns)

    Returns:
        pd.Series with summary stats
    """
    # Geometric mean return (annualized)
    gross_return = (1 + returns).prod()
    years = len(returns) / periods_per_year
    geom_mean_annual = gross_return**(1 / years) - 1

    # Std dev (based on arithmetic mean)
    std_annual = returns.std() * np.sqrt(periods_per_year)

    # Downside std dev
    downside_std = returns[returns < 0].std()
    downside_std_annual = downside_std * np.sqrt(periods_per_year)

    # Ratios using geometric mean
    sharpe = (geom_mean_annual - risk_free_rate * periods_per_year) / std_annual if std_annual > 0 else np.nan
    sortino = (geom_mean_annual - risk_free_rate * periods_per_year) / downside_std_annual if downside_std_annual > 0 else np.nan

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Daily VaR
    var = returns.quantile(confidence)

    return pd.Series({
        "Geometric Mean Return (Annualized)": geom_mean_annual,
        "Annualized Std Dev": std_annual,
        "Sharpe Ratio (Geo)": sharpe,
        "Sortino Ratio (Geo)": sortino,
        "Max Drawdown": max_drawdown,
        f"{int(confidence*100)}% Daily VaR": var
    })

def SummarizeReturns(returns: pd.Series, risk_free_rate: float = 0.0, confidence: float = 0.05, periods_per_year: int = 252) -> pd.Series:
    """
    Summarizes a series of daily returns with annualized performance metrics.

    Parameters:
        returns (pd.Series): Daily return series (arithmetic)
        risk_free_rate (float): Daily risk-free rate (default 0.0)
        confidence (float): For Value at Risk (VaR), default is 5%
        periods_per_year (int): Number of trading periods per year (default 252 for daily)

    Returns:
        pd.Series: Annualized summary statistics
    """

    mean_daily = returns.mean()
    std_daily = returns.std()

    mean_annual = mean_daily * periods_per_year
    std_annual = std_daily * np.sqrt(periods_per_year)

    sharpe = (mean_annual - risk_free_rate * periods_per_year) / std_annual if std_annual > 0 else np.nan

    downside_std_daily = returns[returns < 0].std()
    downside_std_annual = downside_std_daily * np.sqrt(periods_per_year)
    sortino = (mean_annual - risk_free_rate * periods_per_year) / downside_std_annual if downside_std_annual > 0 else np.nan

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    var = returns.quantile(confidence)

    return pd.Series({
        "Annualized Mean": mean_annual,
        "Annualized Std Dev": std_annual,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        f"{int(confidence*100)}% Daily VaR": var
    })

def label_market_swings(df: pd.DataFrame, spy_col: str = 'SPY', window: int = 100, threshold: float = 0.1) -> pd.DataFrame:
    """
    Labels each 100-day partition with 'upswing' and 'downswing' flags based on SPY cumulative returns.

    Parameters:
        df (pd.DataFrame): DataFrame containing return columns
        spy_col (str): Name of SPY return column
        window (int): Partition size
        threshold (float): +/- cumulative return threshold for swing classification

    Returns:
        pd.DataFrame: Original dataframe with 'upswing' and 'downswing' columns
    """
    df = df.copy()
    n = len(df)

    upswing = pd.Series(index=df.index, dtype=int)
    downswing = pd.Series(index=df.index, dtype=int)

    for i in range(0, n, window):
        segment = df.iloc[i:i+window]
        cum_return = (1 + segment[spy_col]).prod() - 1

        upswing_flag = int(cum_return > threshold)
        downswing_flag = int(cum_return < -threshold)

        upswing.iloc[i:i+window] = upswing_flag
        downswing.iloc[i:i+window] = downswing_flag

    df['upswing'] = upswing
    df['downswing'] = downswing

    return df


def rolling_reg(response: pd.Series, regressors: pd.DataFrame, window: int, plot:bool = False)->pd.DataFrame: 
    """
    Takes in a series of returns, and a dataframe of regressors and compute the rolling beta's for each factor in regressors DF

    plot = True will produce a plot of the rolling betas 
    """
    betas = []
    for i in range(window, len(response)): 
        y_window = response.iloc[i - window: i]
        x_window = regressors.iloc[i-window: i]

        X = sm.add_constant(x_window)

        model = sm.OLS(y_window, X).fit()

        betas.append(model.params)
    
    betas_df = pd.DataFrame(betas, index=response.index[window:])

    if plot: 
        fig, ax = plt.subplots(figsize=(10,6))
        betas_df.plot(ax=ax)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    return betas_df

def plot_drawdown(returns:pd.DataFrame)->None: 
    """
    Takes in a date-indexed dataframe with (possibly) multiple returns series and plots the drawdown of each series
    """

    plot_dataframe = pd.DataFrame(index=returns.index)
    for col in returns.columns: 
        cumulative = (1 + returns[col]).cumprod()
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        plot_dataframe[col] = drawdowns
    
    
    fig, ax = plt.subplots(figsize=(10,6))
    for col in plot_dataframe.columns:
        plot_dataframe.loc[:,col].plot(ax=ax, label=col)
        plt.fill_between(plot_dataframe.index, plot_dataframe[col], alpha=0.2)
    
    plt.legend()
    plt.title("Drawdown")
    plt.tight_layout()
    plt.show()