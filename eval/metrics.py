import empyrical as ep
import numpy as np

def make_summary_sharpe(portfolio_list):
    def calculate_daily_returns(portfolio_list):
        daily_returns = []
        for i in range(1, len(portfolio_list)):
            daily_return = (portfolio_list[i] - portfolio_list[i - 1]) / portfolio_list[i - 1]
            daily_returns.append(daily_return)
        return np.array(daily_returns)

    daily_returns = calculate_daily_returns(portfolio_list)
    annual_return = ep.annual_return(daily_returns, period='daily', annualization=2190)
    annual_volatility = ep.annual_volatility(daily_returns, period='daily', annualization=2190)
    sharpe_ratio = ep.sharpe_ratio(daily_returns, period='daily', annualization=2190)
    sortino_ratio = ep.sortino_ratio(daily_returns, period='daily', annualization=2190)
    max_drawdown = ep.max_drawdown(daily_returns)

    print(f"Sum Periods: {len(daily_returns)}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Annual Volatility: {annual_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}")
