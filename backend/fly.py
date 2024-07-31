# from flask import Flask, request, jsonify
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
# import requests
# from bs4 import BeautifulSoup
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize
# import random
# import yfinance as yf


# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Read data from CSV
#     data = pd.read_csv('stock_data.csv', header=[0, 1], index_col=0)

#     # Feature Engineering
#     features = []
#     tickers = data['Close'].columns.tolist()

#     for ticker in tickers:
#         stock_data = data['Close'][ticker].dropna()
#         returns = stock_data.pct_change().dropna() * 252  # Annualized
#         volatilities = returns.rolling(window=30).std().dropna() * np.sqrt(252)
#         moving_avg = stock_data.rolling(window=30).mean().dropna()
#         momentum = returns.rolling(window=30).mean().dropna()
        
#         feature_df = pd.DataFrame({
#             'returns': returns,
#             'volatility': volatilities,
#             'moving_avg': moving_avg,
#             'momentum': momentum
#         }).dropna()
#         feature_df['ticker'] = ticker
#         features.append(feature_df)

#     features_df = pd.concat(features).reset_index(drop=True)

#     # Take input from the request
#     investment_amount = request.json['investment_amount']
#     investment_period = request.json['investment_period']
#     risk_tolerance = request.json['risk_tolerance']
#     expected_return = request.json['expected_return']

#     investment_amount=float(investment_amount)
#     investment_period=float(investment_period)
#     expected_return=float(expected_return)

#     # Convert risk level to quantile
#     risk_quantile = {"low": 0.1, "medium": 0.5, "high": 0.9}.get(risk_tolerance.lower(), 0.5)

#     # Labeling Data based on risk quantile
#     features_df['label'] = features_df.groupby('ticker')['returns'].transform(lambda x: (x > x.quantile(risk_quantile)).astype(int))

#     # Drop rows with any missing values to ensure X and y have the same length
#     features_df.dropna(inplace=True)

#     # Reset index to ensure unique indices
#     features_df.reset_index(drop=True, inplace=True)

#     # Prepare data for model
#     X = features_df.drop(columns=['label', 'ticker'])
#     y = features_df['label']

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     # Model Training
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)

#     # Evaluate the model
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     rmse = mean_squared_error(y_test, y_pred, squared=False)
#     mae = mean_absolute_error(y_test, y_pred)

#     # Prediction: Ranking stocks
#     probs = model.predict_proba(X_test)[:, 1]
#     X_test = X_test.copy()  # Avoid SettingWithCopyWarning
#     X_test['prob'] = probs
#     X_test['ticker'] = features_df.loc[X_test.index, 'ticker'].values

#     top_stocks = X_test.groupby('ticker')['prob'].mean().nlargest(6)
#     top_stock_tickers = top_stocks.index.tolist()

#     # Calculate investment per stock
#     investment_per_stock = investment_amount / len(top_stock_tickers)

#     # Calculate return per stock
#     return_per_stock = expected_return / len(top_stock_tickers)

#     # Calculate investment return
#     investment_return = return_per_stock * investment_per_stock

#     # Calculate investment value after period
#     investment_value_after_period = investment_per_stock + (investment_return * investment_period)

#     selected_stocks = data['Adj Close'][top_stock_tickers]


#     ret = selected_stocks.pct_change().dropna()

#     # Define the function to calculate portfolio statistics
#     def portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate=0.017):
#         weights = np.array(weights)
#         port_return = np.sum(mean_returns * weights) * 252
#         port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
#         sharpe_ratio = (port_return - risk_free_rate) / port_volatility
#         return port_return, port_volatility, sharpe_ratio

#     # Define the function to minimize (negative Sharpe ratio)
#     def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.017):
#         return -portfolio_statistics(weights, mean_returns, cov_matrix, risk_free_rate)[2]

#     # Constraints and bounds
#     noa = len(selected_stocks.columns)
#     cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
#     bounds = tuple((0, 1) for _ in range(noa))
#     initial_guess = noa * [1. / noa]

#     # Calculate mean returns and covariance matrix
#     mean_returns = ret.mean()
#     cov_matrix = ret.cov()

#     # Optimization
#     opt_result = minimize(neg_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix, 0.017),
#                         method='SLSQP', bounds=bounds, constraints=cons)

#     opt_weights = opt_result.x

#     # Portfolio with optimized weights
#     ret['MP'] = ret.dot(opt_weights)
#     opt_port_return, opt_port_volatility, opt_sharpe_ratio = portfolio_statistics(opt_weights, mean_returns, cov_matrix, 0.017)

#     print("Optimized Weights:", opt_weights)
#     print("Optimized Portfolio Sharpe Ratio:", opt_sharpe_ratio)

#     # Function to calculate annualized risk and return
#     def fetch_stock_info(tickers):
#         stock_info = {}
#         for ticker in tickers:
#             stock = yf.Ticker(ticker)
#             info = stock.info
#             stock_info[ticker] = {
#                 "Sector": info.get("sector"),
#                 "Industry": info.get("industry"),
#                 "Summary": info.get("longBusinessSummary"),
#                 "Market Cap": info.get("marketCap"),
#                 "PE Ratio": info.get("trailingPE"),
#                 "Dividend Yield": info.get("dividendYield")
#             }
#         return stock_info

#     def ann_risk_return(returns_df):
#         summary = returns_df.agg(['mean', 'std']).T
#         summary.columns = ["Return", "Risk"]
#         summary.Return = summary.Return * 252
#         summary.Risk = summary.Risk * np.sqrt(252)
#         return summary

#     summary = ann_risk_return(ret)
#     summary['sharpe'] = (summary['Return'] - 0.017) / summary['Risk']
#     print(summary)

#     # Print the percentage weights of the optimized portfolio
#     percentage_weights = opt_weights * 100
#     print("Optimized Portfolio Weights in Percentages:")
#     for stock, weight in zip(selected_stocks.columns, percentage_weights):
#         print(f"{stock}: {weight:.2f}%")
    
#     portfolio_data = {
#         "Optimized_Portfolio_Weights_in_Percentage": []
#     }
#     for stock, weight in zip(selected_stocks.columns, percentage_weights):
#         portfolio_data["Optimized_Portfolio_Weights_in_Percentage"].append({
#             "Ticker": stock,
#             "Weight": f"{weight:.2f}%"
#         })

#     response_dict = {stock: weight for stock, weight in zip(selected_stocks.columns, percentage_weights)}


#     opt_port_return_percentage = opt_port_return * 100
#     print("Optimized Portfolio Annualized Return:", opt_port_return_percentage, "%")

#     # Calculate annualized return and risk in percentage
#     annualized_return_percentage = summary['Return'] * 100
#     annualized_risk_percentage = summary['Risk'] * 100

#     stock_info = fetch_stock_info(top_stock_tickers)


#     # Print the summary with return and risk in percentage
#     print("Annualized Return and Risk in Percentage:")
#     for ticker, ret, risk, sharpe in zip(summary.index, annualized_return_percentage, annualized_risk_percentage, summary['sharpe']):
#         print(f"{ticker}: Return: {ret:.2f}%, Risk: {risk:.2f}%, Sharpe: {sharpe:.2f}")

#     annual_data = {"Annualized_Return_and_Risk": []} 
#     for ticker, ret, risk, sharpe in zip(summary.index, annualized_return_percentage, annualized_risk_percentage, summary['sharpe']):
#         annual_data["Annualized_Return_and_Risk"].append({
#             "Ticker": ticker,
#             "Return": f"{ret:.2f}%",
#             "Risk": f"{risk:.2f}%",
#             "Sharpe": f"{sharpe:.2f}"
#         })  


#     result = {
#         "top_stock_tickers": top_stock_tickers,
#         "investment_per_stock": investment_per_stock,
#         "investment_return": investment_return,
#         "investment_amount": investment_amount,
#         "investment_value_after_period": investment_value_after_period,
#         "accuracy": accuracy,
#         "rmse": rmse,
#         "mae": mae,
#         "optimized_portfolio_sharper_ratio" :opt_sharpe_ratio,
#         "optimized_portfolio_weights_in_percentage" : portfolio_data,
#         "optimized_portfolio_annualized_return" : opt_port_return_percentage,
#         "annualized_return_and_risk_in_percentage" : annual_data,
#         'stock_info' : stock_info,
#     }

#     return jsonify(result)

# def load_and_process_data():
#     # Load data from CSV
#     st = pd.read_csv('sp500_selected_stocks.csv', header=0, index_col=0)

#     # Calculate daily returns
#     daily_returns = st.pct_change().dropna()

#     # Load S&P 500 data
#     sp500 = yf.download(tickers='^GSPC', start='2014-01-01')['Adj Close']
#     daily_returns['^GSPC'] = sp500.pct_change().dropna()

#     # Calculate annualized risk and return
#     def ann_risk_return(returns_df):
#         summary = returns_df.agg(['mean', 'std']).T
#         summary.columns = ["Return", "Risk"]
#         summary.Return = summary.Return * 252
#         summary.Risk = summary.Risk * np.sqrt(252)
#         return summary

#     summary = ann_risk_return(daily_returns)

#     # Calculate Sharpe ratio
#     risk_free_return = 0.017  # 1.7%
#     summary['sharpe'] = (summary['Return'] - risk_free_return) / summary['Risk']
#     summary['TotalRisk_var'] = np.power(summary.Risk, 2)

#     # Calculate covariance matrix
#     COV = daily_returns.cov() * 252
#     summary['SystRisk_var'] = COV.iloc[:, -1]
#     summary['UnsystRisk_var'] = summary['TotalRisk_var'] - summary["SystRisk_var"]

#     # Calculate beta and CAPM return
#     summary['beta'] = summary.SystRisk_var / summary.loc['^GSPC', 'SystRisk_var']
#     summary['capm_ret'] = risk_free_return + (summary.loc['^GSPC', 'Return'] - risk_free_return) * summary.beta
#     summary['alpha'] = summary.Return - summary.capm_ret

#     # Filter result based on alpha > 0 and beta < 1
#     result = summary.loc[(summary.alpha > 0) & (summary.beta < 1)]

#     # Convert DataFrame to dictionary
#     result_dict = summary.to_dict(orient='index')

#     return result_dict

# @app.route('/summary', methods=['GET'])
# def get_summary():
#     summary = load_and_process_data()
#     return jsonify(summary)
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import yfinance as yf


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Read data from CSV
    data = pd.read_csv('stock_data.csv', header=[0, 1], index_col=0)

    # Feature Engineering
    features = []
    tickers = data['Close'].columns.tolist()

    for ticker in tickers:
        stock_data = data['Close'][ticker].dropna()
        returns = stock_data.pct_change() * 252  # Multiply by 252 for annualization
        volatilities = returns.rolling(window=30).std() * np.sqrt(252)  # Adjust for annualization
        moving_avg = stock_data.rolling(window=30).mean()
        momentum = returns.rolling(window=30).mean()

        feature_df = pd.DataFrame({
            'returns': returns,
            'volatility': volatilities,
            'moving_avg': moving_avg,
            'momentum': momentum
        })
        feature_df['ticker'] = ticker
        features.append(feature_df)

    features_df = pd.concat(features)

    # Take input from the request
    investment_amount = request.json['investment_amount']
    investment_period = request.json['investment_period']
    risk_tolerance = request.json['risk_tolerance']
    expected_return = request.json['expected_return']

    investment_amount=float(investment_amount)
    investment_period=float(investment_period)
    expected_return=float(expected_return)

    # Convert risk level to quantile
    risk_quantile = {"low": 0.1, "medium": 0.5, "high": 0.9}.get(risk_tolerance.lower(), 0.5)

    # Labeling Data based on risk quantile
    features_df['label'] = features_df.groupby('ticker')['returns'].transform(lambda x: (x > x.quantile(risk_quantile)).astype(int))

    # Drop rows with any missing values to ensure X and y have the same length
    features_df.dropna(inplace=True)

    # Reset index to ensure unique indices
    features_df.reset_index(drop=True, inplace=True)

    # Prepare data for model
    X = features_df.drop(columns=['label', 'ticker'])
    y = features_df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    # Prediction: Ranking stocks
    probs = model.predict_proba(X_test)[:, 1]
    X_test = X_test.copy()  # Avoid SettingWithCopyWarning
    X_test['prob'] = probs
    X_test['ticker'] = features_df.loc[X_test.index, 'ticker'].values

    top_stocks = X_test.groupby('ticker')['prob'].mean().nlargest(7)
    top_stock_tickers = top_stocks.index.tolist()

    # Calculate investment per stock
    investment_per_stock = investment_amount / len(top_stock_tickers)

    # Calculate return per stock
    return_per_stock = expected_return / len(top_stock_tickers)

    # Calculate investment return
    investment_return = return_per_stock * investment_per_stock

    # Calculate investment value after period
    investment_value_after_period = investment_per_stock + (investment_return * investment_period)

    selected_stocks = data['Adj Close'][top_stock_tickers]


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
    def fetch_stock_info(tickers):
        stock_info = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            stock_info[ticker] = {
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
                "Summary": info.get("longBusinessSummary"),
                "Market_Cap": info.get("marketCap"),
                "PE_Ratio": info.get("trailingPE"),
                "Divident_Yield": info.get("dividendYield")
            }
        return stock_info

    def ann_risk_return(returns_df):
        summary = returns_df.agg(['mean', 'std']).T
        summary.columns = ["Return", "Risk"]
        summary.Return = summary.Return * 252
        summary.Risk = summary.Risk * np.sqrt(252)
        return summary

    summary = ann_risk_return(ret)
    summary['sharpe'] = (summary['Return'] - 0.017) / summary['Risk']
    print(summary)

    # Print the percentage weights of the optimized portfolio
    percentage_weights = opt_weights * 100
    print("Optimized Portfolio Weights in Percentages:")
    for stock, weight in zip(selected_stocks.columns, percentage_weights):
        print(f"{stock}: {weight:.2f}%")
    
    portfolio_data = {
        "Optimized_Portfolio_Weights_in_Percentage": []
    }
    for stock, weight in zip(selected_stocks.columns, percentage_weights):
        portfolio_data["Optimized_Portfolio_Weights_in_Percentage"].append({
            "Ticker": stock,
            "Weight": f"{weight:.2f}%"
        })

    response_dict = {stock: weight for stock, weight in zip(selected_stocks.columns, percentage_weights)}


    opt_port_return_percentage = opt_port_return * 100
    print("Optimized Portfolio Annualized Return:", opt_port_return_percentage, "%")

    # Calculate annualized return and risk in percentage
    annualized_return_percentage = summary['Return'] * 100
    annualized_risk_percentage = summary['Risk'] * 100

    stock_info = fetch_stock_info(top_stock_tickers)


    # Print the summary with return and risk in percentage
    print("Annualized Return and Risk in Percentage:")
    for ticker, ret, risk, sharpe in zip(summary.index, annualized_return_percentage, annualized_risk_percentage, summary['sharpe']):
        print(f"{ticker}: Return: {ret:.2f}%, Risk: {risk:.2f}%, Sharpe: {sharpe:.2f}")

    annual_data = {"Annualized_Return_and_Risk": []} 
    for ticker, ret, risk, sharpe in zip(summary.index, annualized_return_percentage, annualized_risk_percentage, summary['sharpe']):
        annual_data["Annualized_Return_and_Risk"].append({
            "Ticker": ticker,
            "Return": f"{ret:.2f}%",
            "Risk": f"{risk:.2f}%",
            "Sharpe": f"{sharpe:.2f}"
        })  


    result = {
        "top_stock_tickers": top_stock_tickers,
        "investment_per_stock": investment_per_stock,
        "investment_return": investment_return,
        "investment_amount": investment_amount,
        "investment_value_after_period": investment_value_after_period,
        "accuracy": accuracy,
        "rmse": rmse,
        "mae": mae,
        "optimized_portfolio_sharper_ratio" :opt_sharpe_ratio,
        "optimized_portfolio_weights_in_percentage" : portfolio_data,
        "optimized_portfolio_annualized_return" : opt_port_return_percentage,
        "annualized_return_and_risk_in_percentage" : annual_data,
        'stock_info' : stock_info,
    }

    return jsonify(result)

def load_and_process_data():
    # Load data from CSV
    st = pd.read_csv('sp500_selected_stocks.csv', header=0, index_col=0)

    # Calculate daily returns
    daily_returns = st.pct_change().dropna()

    # Load S&P 500 data
    sp500 = yf.download(tickers='^GSPC', start='2014-01-01')['Adj Close']
    daily_returns['^GSPC'] = sp500.pct_change().dropna()

    # Calculate annualized risk and return
    def ann_risk_return(returns_df):
        summary = returns_df.agg(['mean', 'std']).T
        summary.columns = ["Return", "Risk"]
        summary.Return = summary.Return * 252
        summary.Risk = summary.Risk * np.sqrt(252)
        return summary

    summary = ann_risk_return(daily_returns)

    # Calculate Sharpe ratio
    risk_free_return = 0.017  # 1.7%
    summary['sharpe'] = (summary['Return'] - risk_free_return) / summary['Risk']
    summary['TotalRisk_var'] = np.power(summary.Risk, 2)

    # Calculate covariance matrix
    COV = daily_returns.cov() * 252
    summary['SystRisk_var'] = COV.iloc[:, -1]
    summary['UnsystRisk_var'] = summary['TotalRisk_var'] - summary["SystRisk_var"]

    # Calculate beta and CAPM return
    summary['beta'] = summary.SystRisk_var / summary.loc['^GSPC', 'SystRisk_var']
    summary['capm_ret'] = risk_free_return + (summary.loc['^GSPC', 'Return'] - risk_free_return) * summary.beta
    summary['alpha'] = summary.Return - summary.capm_ret

    # Filter result based on alpha > 0 and beta < 1
    result = summary.loc[(summary.alpha > 0) & (summary.beta < 1)]

    # Convert DataFrame to dictionary
    result_dict = summary.to_dict(orient='index')

    return result_dict

@app.route('/summary', methods=['GET'])
def get_summary():
    summary = load_and_process_data()
    return jsonify(summary)
if __name__ == '__main__':
    app.run(debug=True)