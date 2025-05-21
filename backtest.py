import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import math

def get_rfr(observation_start, observation_end):
    fred_api_key = '48ceac4c50be275b61c7a6445466d179'
    series_id = 'DGS10'  # US 10-year Treasury yield

    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&api_key={fred_api_key}"
        f"&file_type=json"
        f"&observation_start={observation_start}"
        f"&observation_end={observation_end}"
    )

    response = requests.get(url)
    data = response.json()

    if 'observations' in data:
        observations = data['observations']
        # Create a time series DataFrame only for valid yield values
        time_series = {obs['date']: obs['value'] for obs in observations if obs['value'] != '.'}
        df = pd.DataFrame(list(time_series.items()), columns=['date', 'yield'])
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        return df
    else:
        print("Error fetching data:", data)
        return pd.DataFrame()

class BacktestEngine:
    def __init__(self, df, spy_data, fama_french_factors=None, vol_data=None):
        """
        Parameters:
            df: DataFrame containing your strategy data; must include a 'date' column and a 'daily_return' column.
            spy_data: DataFrame containing SPY market data. This DataFrame must have a 'Date' column and an 'Adj Close' column.
            fama_french_factors: (Optional) DataFrame of Fama-French factors.
            vol_data: (Optional) DataFrame of volatility data.
        """
        self.df = df.copy()
        self.spy_data = spy_data.copy()
        self.fama_french_factors = fama_french_factors
        self.vol_data = vol_data
        self.benchmark_ticker = 'SPY'
        
        # Prepare the strategy DataFrame
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.sort_values('date', inplace=True)
        if 'daily_return' not in self.df.columns:
            raise ValueError("DataFrame must contain a 'daily_return' column.")
        
        # Prepare the benchmark (SPY) DataFrame
        self.spy_data['Date'] = pd.to_datetime(self.spy_data['Date'])
        self.spy_data.sort_values('Date', inplace=True)
        # Rename the date column to 'date' for consistency
        self.spy_data.rename(columns={'Date': 'date'}, inplace=True)
        benchmark = self.spy_data.copy()
        benchmark['benchmark_return'] = benchmark['Adj Close'].pct_change().fillna(0)
        
        # Define the start and end dates based on your strategy data
        start = self.df['date'].min()
        end = self.df['date'].max()
        
        # Get risk-free rate data from FRED and compute daily risk-free rate
        rfr_df = get_rfr(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        rfr_df['daily_rf'] = pd.to_numeric(rfr_df['yield'], errors='coerce') / 100 / 252
        
        # Merge your strategy data with benchmark and risk-free rate data
        merged = pd.merge(self.df, benchmark[['date', 'benchmark_return']], on='date', how='inner')
        merged = pd.merge(merged, rfr_df[['date', 'daily_rf']], on='date', how='inner')
        
        # Calculate excess returns
        merged['excess_return'] = merged['daily_return'] - merged['daily_rf']
        merged['excess_bench']  = merged['benchmark_return'] - merged['daily_rf']
        
        # Store the merged DataFrame and compute all metrics
        self.merged = merged
        self._compute_core_metrics()
        self._compute_fama_french_regression()
        self._compute_vol_regression()
        self._compute_calmar_ratio()
        self._compute_beta()
        self._compute_treynor_ratio()
        #self._compute_alpha()
        self._compute_standard_beta()
        self._compute_omega_ratio()
        self._compute_var_cvar()
        self._compute_max_drawdown_duration()
        self._compute_profit_factor()
        self._compute_annual_metrics()
        self._compute_recovery_factor()

    def _compute_core_metrics(self):
        excess_return = self.merged['excess_return']
        bench_return  = self.merged['benchmark_return']
        daily_return  = self.merged['daily_return']
        
        sharpe = np.nan
        sortino = np.nan
        info_ratio = np.nan
        downside_beta = np.nan

        if excess_return.std() != 0:
            sharpe = np.sqrt(252) * excess_return.mean() / excess_return.std()

        negative_excess = excess_return[excess_return < 0]
        if negative_excess.std() != 0:
            sortino = np.sqrt(252) * excess_return.mean() / negative_excess.std()

        diff = daily_return - bench_return
        if diff.std() != 0:
            info_ratio = np.sqrt(252) * diff.mean() / diff.std()

        dn = self.merged[self.merged['benchmark_return'] < 0]
        if len(dn) > 1:
            cov_down = np.cov(dn['excess_return'], dn['excess_bench'])[0, 1]
            var_down = np.var(dn['excess_bench'])
            if var_down != 0:
                downside_beta = cov_down / var_down

        self.merged['strategy_cum'] = (1 + self.merged['daily_return']).cumprod()
        roll_max = self.merged['strategy_cum'].cummax()
        drawdown = self.merged['strategy_cum'] / roll_max - 1
        max_dd = drawdown.min()

        self.sharpe = sharpe
        self.sortino = sortino
        self.info_ratio = info_ratio
        self.downside_beta = downside_beta
        self.max_dd = max_dd

    def _compute_fama_french_regression(self):
        self.regression_results = None
        if self.fama_french_factors is not None:
            factors = self.fama_french_factors.copy()
            factors['date'] = pd.to_datetime(factors['date'])
            merged_factors = pd.merge(self.merged[['date', 'excess_return']], 
                                      factors, on='date', how='inner')
            merged_factors.dropna(inplace=True)
            y = merged_factors['excess_return']
            X = merged_factors.drop(['date', 'excess_return'], axis=1)
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            self.regression_results = model.summary()

    def _compute_vol_regression(self):
        self.vol_regression_results = None
        if self.vol_data is not None:
            vol = self.vol_data[['date', 'close']].copy()
            vol['date'] = pd.to_datetime(vol['date'])
            merged_factors = pd.merge(self.merged[['date', 'excess_return']], 
                                      vol, on='date', how='inner')
            merged_factors.dropna(inplace=True)
            y = merged_factors['excess_return']
            X = merged_factors.drop(['date', 'excess_return'], axis=1)
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            self.vol_regression_results = model.summary()

    def _compute_calmar_ratio(self):
        if self.max_dd != 0:
            annual_return = self.merged['daily_return'].mean() * 252
            self.calmar_ratio = annual_return / abs(self.max_dd)
        else:
            self.calmar_ratio = np.nan

    def _compute_beta(self):
        benchmark_return = self.merged['benchmark_return']
        daily_return = self.merged['daily_return']
        covariance = np.cov(daily_return, benchmark_return)[0, 1]
        variance = np.var(benchmark_return)
        self.beta = covariance / variance if variance != 0 else np.nan

    def _compute_treynor_ratio(self):
        if self.beta != 0:
            annual_return = self.merged['daily_return'].mean() * 252
            risk_free_rate = self.merged['daily_rf'].mean() * 252
            self.treynor_ratio = (annual_return - risk_free_rate) / self.beta
        else:
            self.treynor_ratio = np.nan

    #def _compute_alpha(self):
    #    if self.regression_results:
    #        self.alpha = self.regression_results.params['const']
    #    else:
    #        self.alpha = np.nan

    def _compute_standard_beta(self):
        benchmark_return = self.merged['benchmark_return']
        daily_return = self.merged['daily_return']
        covariance = np.cov(daily_return, benchmark_return)[0, 1]
        variance = np.var(benchmark_return)
        self.standard_beta = covariance / variance if variance != 0 else np.nan

    def _compute_omega_ratio(self, threshold=0):
        gains = self.merged['daily_return'][self.merged['daily_return'] > threshold] - threshold
        losses = threshold - self.merged['daily_return'][self.merged['daily_return'] < threshold]
        gain_sum = gains.sum()
        loss_sum = losses.sum()
        self.omega_ratio = gain_sum / loss_sum if loss_sum != 0 else np.nan

    def _compute_var_cvar(self, confidence_level=0.95):
        var = np.percentile(self.merged['daily_return'], (1 - confidence_level) * 100)
        cvar = self.merged['daily_return'][self.merged['daily_return'] <= var].mean()
        self.var = var
        self.cvar = cvar

    def _compute_max_drawdown_duration(self):
        self.merged['strategy_cum'] = (1 + self.merged['daily_return']).cumprod()
        roll_max = self.merged['strategy_cum'].cummax()
        drawdown = self.merged['strategy_cum'] / roll_max - 1
        drawdown_period = (drawdown != 0).astype(int)
        drawdown_diff = drawdown_period.diff()
        drawdown_starts = drawdown_diff[drawdown_diff == 1].index
        drawdown_ends = drawdown_diff[drawdown_diff == -1].index

        if len(drawdown_starts) == 0:
            self.max_dd_duration = 0
        else:
            if len(drawdown_ends) < len(drawdown_starts):
                drawdown_ends = drawdown_ends.append(pd.Index([len(drawdown_period) - 1]))
            durations = drawdown_ends - drawdown_starts
            self.max_dd_duration = durations.max()

    def _compute_profit_factor(self):
        gross_profit = self.merged['daily_return'][self.merged['daily_return'] > 0].sum()
        gross_loss = -self.merged['daily_return'][self.merged['daily_return'] < 0].sum()
        self.profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

    def _compute_annual_metrics(self):
        cumulative_return = (1 + self.merged['daily_return']).prod() - 1
        num_years = (self.merged['date'].max() - self.merged['date'].min()).days / 365.25
        self.annualized_return = (1 + cumulative_return) ** (1 / num_years) - 1
        self.annualized_volatility = self.merged['daily_return'].std() * np.sqrt(252)

    def _compute_recovery_factor(self):
        cumulative_return = (1 + self.merged['daily_return']).prod() - 1
        self.recovery_factor = cumulative_return / abs(self.max_dd) if self.max_dd != 0 else np.nan

    def get_sharpe(self):
        return self.sharpe

    def get_sortino(self):
        return self.sortino

    def get_information_ratio(self):
        return self.info_ratio

    def get_downside_beta(self):
        return self.downside_beta

    def get_max_drawdown(self):
        return self.max_dd

    def get_fama_french_regression(self):
        return self.regression_results
    
    def get_vol_regression(self):
        return self.vol_regression_results
    
    def get_calmar_ratio(self):
        return self.calmar_ratio

    def get_beta(self):
        return self.beta

    def get_treynor_ratio(self):
        return self.treynor_ratio

    #def get_alpha(self):
    #    return self.alpha

    def get_standard_beta(self):
        return self.standard_beta

    def get_omega_ratio(self):
        return self.omega_ratio

    def get_var(self):
        return self.var

    def get_cvar(self):
        return self.cvar

    def get_max_drawdown_duration(self):
        return self.max_dd_duration

    def get_profit_factor(self):
        return self.profit_factor

    def get_annualized_return(self):
        return self.annualized_return

    def get_annualized_volatility(self):
        return self.annualized_volatility

    def get_recovery_factor(self):
        return self.recovery_factor

    def get_all_metrics(self):
        return {
            'Annualized Sharpe': self.get_sharpe(),
            'Annualized Sortino': self.get_sortino(),
            'Information Ratio': self.get_information_ratio(),
            'Downside Beta': self.get_downside_beta(),
            'Max Drawdown': self.get_max_drawdown(),
            'Calmar Ratio': self.get_calmar_ratio(),
            'Treynor Ratio': self.get_treynor_ratio(),
            #'Alpha': self.get_alpha(),
            'Standard Beta': self.get_standard_beta(),
            'Omega Ratio': self.get_omega_ratio(),
            'Value at Risk (VaR)': self.get_var(),
            'Conditional Value at Risk (CVaR)': self.get_cvar(),
            'Max Drawdown Duration': self.get_max_drawdown_duration(),
            'Profit Factor': self.get_profit_factor(),
            'Annualized Return': self.get_annualized_return(),
            'Annualized Volatility': self.get_annualized_volatility(),
            'Recovery Factor': self.get_recovery_factor(),
            'Fama-French Regression': self.get_fama_french_regression(),
            'Volatility Regression': self.get_vol_regression()
        }

    def get_core_metrics(self):
        """Returns all core computed performance metrics in a single dictionary."""
        return {
            'Annualized Sharpe': self.get_sharpe(),
            'Annualized Sortino': self.get_sortino(),
            'Information Ratio': self.get_information_ratio(),
            'Downside Beta': self.get_downside_beta(),
            'Max Drawdown': self.get_max_drawdown(),
            'Fama-French Regression': self.get_fama_french_regression(),
            'Volatility Regression': self.get_vol_regression()
        }

    def plot_strategy(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.merged['date'], self.merged['strategy_cum'], label='Strategy')
        bench_cum = (1 + self.merged['benchmark_return']).cumprod()
        plt.plot(self.merged['date'], bench_cum, label=self.benchmark_ticker)
        plt.legend()
        plt.show()

    def plot_svol_vs_strategy_vol(self, window=30):
        self.vol_data['date'] = pd.to_datetime(self.vol_data['date'])
        self.vol_data = self.vol_data[['date', 'close']].copy()
        self.vol_data['svol_return'] = self.vol_data['close'].pct_change().fillna(0)

        merged_vol = pd.merge(self.merged[['date', 'excess_return']], self.vol_data, on='date', how='inner')
        merged_vol.loc[abs(merged_vol['excess_return']) < 0.0001, 'excess_return'] = np.nan
        merged_vol['strategy_vol'] = merged_vol['excess_return'].rolling(window).std()
        
        plt.figure(figsize=(10, 6))
        plt.plot(merged_vol['date'], merged_vol['strategy_vol'], label='Strategy Volatility', color='blue')
        plt.plot(merged_vol['date'], merged_vol['svol_return'], label='SVOL Return', color='red', alpha=0.7)
        plt.legend()
        plt.title(f'Strategy Volatility vs. SVOL ({window}-Day Rolling)')
        plt.show()

    # --- New Methods Added ---
    def plot_rolling_volatility(self, window=30, title="30-Day Rolling Volatility"):
        df = self.merged.copy()
        df['rolling_vol'] = df['daily_return'].rolling(window).std() * np.sqrt(252)
        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['rolling_vol'], label=f'{window}-Day Rolling Volatility')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_drawdown(self, title="Drawdown Chart"):
        df = self.merged.copy()
        df['cum_return'] = (1 + df['daily_return']).cumprod()
        df['rolling_max'] = df['cum_return'].cummax()
        df['drawdown'] = df['cum_return'] / df['rolling_max'] - 1
        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['drawdown'], label='Drawdown', color='red')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_return_distribution(self, title="Return Distribution"):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.merged['daily_return'], bins=50, kde=True, color='blue')
        plt.title(title)
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.show()

class ComparisonBacktest:
    def __init__(self, df, return_cols):
        """
        df : DataFrame with 'date' + multiple return columns (daily returns)
        return_cols : list of column‐names in df to compare
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.set_index('date', inplace=True)
        self.returns = self.df[return_cols].fillna(0)

        # cumulative returns
        self.cum_returns = (1 + self.returns).cumprod()

        # drawdowns
        self.rolling_max = self.cum_returns.cummax()
        self.drawdowns = self.cum_returns / self.rolling_max - 1

    def plot_cumulative_returns(self, figsize=(12,6), title="Cumulative Returns"):
        plt.figure(figsize=figsize)
        for col in self.cum_returns:
            plt.plot(self.cum_returns.index, self.cum_returns[col], label=col)
        plt.legend()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.tight_layout()
        plt.show()

    def plot_drawdowns(self, figsize=(12,6), title="Drawdown"):
        plt.figure(figsize=figsize)
        for col in self.drawdowns:
            plt.plot(self.drawdowns.index, self.drawdowns[col], label=col)
        plt.legend()
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.tight_layout()
        plt.show()

    def plot_return_distributions(
        self,
        bins=50,
        figsize=(12,8),
        cols_per_row=2,
        n_xticks=5
    ):
        rets = self.returns

        # 1) pooled mean across all returns
        overall_mean = rets.stack().mean()

        # 2) global bin edges and y‐limit
        min_ret, max_ret = rets.min().min(), rets.max().max()
        bins_edges = np.linspace(min_ret, max_ret, bins + 1)
        max_count = max(
            np.histogram(rets[col], bins=bins_edges)[0].max()
            for col in rets.columns
        )

        # 3) subplot layout
        n = len(rets.columns)
        ncols = cols_per_row
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            sharex=False, sharey=True
        )
        axes = axes.flatten()

        cmap = plt.get_cmap("tab10")
        xticks = np.linspace(min_ret, max_ret, n_xticks)

        # 4) draw each histogram + the same overall‐mean line
        for idx, col in enumerate(rets.columns):
            ax = axes[idx]
            color = cmap(idx % cmap.N)

            ax.hist(rets[col], bins=bins_edges, color=color)
            ax.axvline(
                overall_mean,
                color='black',
                linestyle='--',
                linewidth=1.5,
                label=f'Overall mean ({overall_mean:.4%})'
            )

            ax.set_title(col, fontsize=10)
            ax.set_xlim(min_ret, max_ret)
            ax.set_ylim(0, max_count * 1.05)
            ax.set_xticks(xticks)
            ax.set_xlabel("Daily Return")
            ax.set_ylabel("Frequency")
            ax.legend(loc='upper right', fontsize=8)

        # 5) hide any empty subplots
        for ax in axes[n:]:
            ax.set_visible(False)

        # 6) super‐title + layout
        fig.suptitle("Return Distributions", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

