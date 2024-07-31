import joblib
import numpy as np
import warnings
from flask import Flask, request, jsonify
import pickle
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

# warnings.simplefilter("ignore", InconsistentVersionWarning)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract JSON data from the request body
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        risk_tolerance = data.get('risk_tolerance')
        investment_period = data.get('investment_period')
        investment_amount = data.get('investment_amount')
        expected_return = data.get('expected_return')

        # Define stocks manually as provided
        stocks = [
          "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", "LT.NS",
    "MARUTI.NS", "AXISBANK.NS", "ITC.NS", "BAJFINANCE.NS", "ASIANPAINT.NS",
    "HCLTECH.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS", "TECHM.NS", "TITAN.NS",
    "M&M.NS", "DRREDDY.NS", "WIPRO.NS", "NESTLEIND.NS", "BPCL.NS",
    "POWERGRID.NS", "ONGC.NS", "NTPC.NS", "COALINDIA.NS", "ADANIPORTS.NS",
    "BAJAJFINSV.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "BRITANNIA.NS", "INDUSINDBK.NS",
    "IOC.NS", "CIPLA.NS", "VEDL.NS", "SHREECEM.NS", "GRASIM.NS",
    "DIVISLAB.NS", "EICHERMOT.NS", "ADANIGREEN.NS", "TATAMOTORS.NS", "DABUR.NS",
    
        ]

        # Load data for selected stocks
        stocks_data = yf.download(stocks, start='2014-01-01')['Adj Close']

        # Calculate daily returns
        daily_returns = stocks_data.pct_change().dropna()

        # Function to calculate annualized return and risk
        def calculate_annualized_metrics(returns):
            annualized_return = returns.mean() * 252
            annualized_risk = returns.std() * np.sqrt(252)
            return annualized_return, annualized_risk

        # Calculate annualized return, risk, and Sharpe ratio for each stock
        stock_metrics = []
        from fredapi import Fred

        # Replace 'YOUR_API_KEY' with your actual FRED API key
        fred = Fred(api_key='b73f7a1a09f317efceeec4829a0301c1')

        try:
            # Retrieve the data for the 10-Year Treasury Constant Maturity Rate
            risk_free_rate_series = fred.get_series('GS10') / 100

            # Get the most recent risk-free rate
            risk_free_rate = risk_free_rate_series.iloc[-1]
            print(f"Latest risk-free rate (10-Year Treasury): {risk_free_rate:.4%}")

        except Exception as e:
            print(f"An error occurred: {e}")

        for stock in daily_returns.columns:
            ann_return, ann_risk = calculate_annualized_metrics(daily_returns[stock])
            sharpe_ratio = (ann_return - risk_free_rate) / ann_risk
            stock_metrics.append((stock, ann_return, ann_risk, sharpe_ratio))

        # Convert to DataFrame
        stock_metrics_df = pd.DataFrame(stock_metrics, columns=['Stock', 'Return', 'Risk', 'Sharpe'])

        # Define risk categories
        def define_risk_categories(df):
            low_risk_threshold = df['Risk'].quantile(0.33)
            high_risk_threshold = df['Risk'].quantile(0.67)
            df['RiskCategory'] = df['Risk'].apply(lambda x: 'Low' if x <= low_risk_threshold else ('High' if x >= high_risk_threshold else 'Medium'))
            return df

        stock_metrics_df = define_risk_categories(stock_metrics_df)

        # Function to select stocks based on user-defined risk preference, Sharpe ratio, and correlation
        def select_stocks(df, returns, num_stocks, risk_preference):
            # Filter by risk preference
            df = df[df['RiskCategory'] == risk_preference]

            # Sort by Sharpe ratio
            df = df.sort_values(by='Sharpe', ascending=False)

            # Calculate correlation matrix
            corr_matrix = returns.corr()

            selected_stocks = []
            for stock in df['Stock']:
                if len(selected_stocks) < num_stocks:
                    add_stock = True
                    for selected_stock in selected_stocks:
                        if corr_matrix[stock][selected_stock] > 0.5:
                            add_stock = False
                            break
                    if add_stock:
                        selected_stocks.append(stock)
            return selected_stocks

        # Get user input for risk preference
        risk_preference = risk_tolerance

        # Select top 6 stocks based on user input
        top_6_stocks = select_stocks(stock_metrics_df, daily_returns, 6, risk_preference)
        selected_stocks = stocks_data[top_6_stocks]

        # Use your existing code to optimize the portfolio for these 6 stocks
        ret = selected_stocks.pct_change().dropna()

        # Define the function to calculate portfolio statistics
        def portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate=0.017):
            weights = np.array(weights)
            port_return = np.sum(mean_returns * weights) * 252
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe_ratio = (port_return - risk_free_rate) / port_volatility
            return port_return, port_volatility, sharpe_ratio

        # Define the function to minimize (negative Sharpe ratio)
        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.017):
            return -portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate)[2]

        # Constraints and bounds
        noa = len(selected_stocks.columns)
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(noa))
        initial_guess = noa * [1. / noa]

        # Calculate mean returns and covariance matrix
        mean_returns = ret.mean()
        cov_matrix = ret.cov()

        # Optimization
        opt_result = minimize(neg_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix, 0.017),
                              method='SLSQP', bounds=bounds, constraints=cons)

        opt_weights = opt_result.x

        # Portfolio with optimized weights
        ret['MP'] = ret.dot(opt_weights)
        opt_port_return, opt_port_volatility, opt_sharpe_ratio = portfolio_statistics(opt_weights, mean_returns, cov_matrix, 0.017)

        print("Optimized Weights:", opt_weights)
        print("Optimized Portfolio Sharpe Ratio:", opt_sharpe_ratio)

        # Function to calculate annualized risk and return
        def ann_risk_return(returns_df):
            summary = returns_df.agg(['mean', 'std']).T
            summary.columns = ["Return", "Risk"]
            summary.Return = summary.Return * 252
            summary.Risk = summary.Risk * np.sqrt(252)
            return summary

        summary = ann_risk_return(ret)
        summary['sharpe'] = (summary['Return'] - 0.017) / summary['Risk']
        print(summary)

        # Plot the results
        plt.figure(figsize=(15, 8))
        plt.scatter(summary.loc[:, "Risk"], summary.loc[:, "Return"], s=20,
                    c=summary.loc[:, "sharpe"], cmap="coolwarm", alpha=0.8)
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(summary.loc[:, "Risk"], summary.loc[:, "Return"], s=50, marker="D",
                    c=summary.loc[:, "sharpe"], cmap="coolwarm")
        plt.scatter(summary.loc["MP", "Risk"], summary.loc["MP", "Return"], s=500, c="black", marker="*")
        plt.annotate("Max SR Portfolio", xy=(summary.loc["MP", "Risk"]-0.02, summary.loc["MP", "Return"]+0.02), size=20, color="black")
        plt.xlabel("Annual Risk (std)", fontsize=15)
        plt.ylabel("Annual Return", fontsize=15)
        plt.title("The Max Sharpe Ratio Portfolio", fontsize=20)
        plt.show()

        # Print the percentage weights of the optimized portfolio
        percentage_weights = opt_weights * 100
        print("Optimized Portfolio Weights in Percentages:")
        for stock, weight in zip(selected_stocks.columns, percentage_weights):
            print(f"{stock}: {weight:.2f}%")

        opt_port_return_percentage = opt_port_return * 100
        print("Optimized Portfolio Annualized Return:", opt_port_return_percentage, "%")

        # Return the prediction result as a JSON response

        # Calculate annualized return and risk in percentage
        annualized_return_percentage = summary['Return'] * 100
        annualized_risk_percentage = summary['Risk'] * 100

        # Print the summary with return and risk in percentage
        print("Annualized Return and Risk in Percentage:")
        for ticker, ret, risk, sharpe in zip(summary.index, annualized_return_percentage, annualized_risk_percentage, summary['sharpe']):
            print(f"{ticker}: Return: {ret:.2f}%, Risk: {risk:.2f}%, Sharpe: {sharpe:.2f}")

        # Print the percentage weights of the optimized portfolio
        percentage_weights = opt_weights * 100
        print("\nOptimized Portfolio Weights in Percentages:")
        for stock, weight in zip(selected_stocks.columns, percentage_weights):
            print(f"{stock}: {weight:.2f}%")


            # Calculate annualized return and risk in percentage
            annualized_return_percentage = summary['Return'] * 100
            annualized_risk_percentage = summary['Risk'] * 100

            # Print the summary with return and risk in percentage
            print("Annualized Return and Risk in Percentage:")
            for ticker, ret, risk, sharpe in zip(summary.index, annualized_return_percentage, annualized_risk_percentage, summary['sharpe']):
                print(f"{ticker}: Return: {ret:.2f}%, Risk: {risk:.2f}%, Sharpe: {sharpe:.2f}")

            # Print the percentage weights of the optimized portfolio
            percentage_weights = opt_weights * 100
            print("\nOptimized Portfolio Weights in Percentages:")
            for stock, weight in zip(selected_stocks.columns, percentage_weights):
                print(f"{stock}: {weight:.2f}%")

            # Calculate and print the percentage of return and risk for the optimized portfolio
            opt_port_return_percentage = opt_port_return * 100
            opt_port_risk_percentage = opt_port_volatility * 100
            print("\nOptimized Portfolio Annualized Return:", opt_port_return_percentage, "%")
            print("Optimized Portfolio Annualized Risk:", opt_port_risk_percentage, "%")

        # Calculate and print the percentage of return and risk for the optimized portfolio
        opt_port_return_percentage = opt_port_return * 100
        opt_port_risk_percentage = opt_port_volatility * 100
        print("\nOptimized Portfolio Annualized Return:", opt_port_return_percentage, "%")
        print("Optimized Portfolio Annualized Risk:", opt_port_risk_percentage, "%")







        return jsonify({
            'optimized_weights': {stock: weight for stock, weight in zip(selected_stocks.columns, percentage_weights)},
            'optimized_portfolio_return': opt_port_return_percentage,
            'optimized_portfolio_volatility': opt_port_volatility,
            'optimized_sharpe_ratio': opt_sharpe_ratio
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



