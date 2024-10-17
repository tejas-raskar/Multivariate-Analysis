import yfinance as yf
import pandas as pd
import pandas_datareader as data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st

def fetch_stock_data(ticker, start_date, end_date):
    yf.pdr_override()
    df = data.data.get_data_yahoo(ticker, start_date, end_date)
    df.index = pd.to_datetime(df.index)  # Ensure the index is a datetime object
    return df

def prepare_data(df):
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    features = ['Open', 'High', 'Low', 'SMA_50', 'SMA_200']
    targets = ['Close', 'Volume', 'Volatility']
    
    X = df[features].dropna()
    y = df[targets].loc[X.index]
    
    return X, y

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=100)),
        'SVM': MultiOutputRegressor(SVR()),
        'Neural Network': MultiOutputRegressor(MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500))
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
    
    return models, scaler, X_test_scaled, y_test

def evaluate_models(models, X_test_scaled, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        r2 = r2_score(y_test, y_pred, multioutput='raw_values')
        results[name] = {'y_pred': y_pred, 'mse': mse, 'r2': r2}
    return results

def plot_results(y_test, results, target_names):
    fig, axes = plt.subplots(len(target_names), 1, figsize=(12, 6*len(target_names)))
    colors = ['b', 'g', 'r', 'c']
    
    # Sort the test set and predictions by date
    y_test_sorted = y_test.sort_index()
    
    for i, (ax, name) in enumerate(zip(axes, target_names)):
        ax.plot(y_test_sorted.index, y_test_sorted.iloc[:, i], label='Actual', color='k', linewidth=2)
        for (model_name, result), color in zip(results.items(), colors):
            pred_df = pd.DataFrame(result['y_pred'], index=y_test.index, columns=target_names).sort_index()
            ax.plot(pred_df.index, pred_df.iloc[:, i], label=f'{model_name} Predicted', color=color, alpha=0.7)
        ax.set_title(f'{name} - Actual vs Predicted')
        ax.set_xlabel('Date')
        ax.set_ylabel(name)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig

def plot_model_comparison(results, target_names):
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    mse_data = [result['mse'] for result in results.values()]
    r2_data = [result['r2'] for result in results.values()]
    
    axes[0].bar(range(len(results)), [m[0] for m in mse_data], align='center')
    axes[0].set_title('MSE Comparison - Closing Price')
    axes[0].set_xticks(range(len(results)))
    axes[0].set_xticklabels(results.keys(), rotation=45)
    axes[0].set_ylabel('Mean Squared Error')
    
    axes[1].bar(range(len(results)), [r[0] for r in r2_data], align='center')
    axes[1].set_title('R-squared Comparison - Closing Price')
    axes[1].set_xticks(range(len(results)))
    axes[1].set_xticklabels(results.keys(), rotation=45)
    axes[1].set_ylabel('R-squared Score')
    
    plt.tight_layout()
    return fig

def main():
    st.title('Stock Price Prediction using Multiple Multivariate Regression Models')
    
    ticker = st.text_input('Enter stock ticker (e.g., AAPL, GOOGL)', 'AAPL')
    start_date = st.date_input('Start date', pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End date', pd.to_datetime('2023-12-31'))
    
    if st.button('Predict'):
        df = fetch_stock_data(ticker, start_date, end_date)
        X, y = prepare_data(df)
        models, scaler, X_test_scaled, y_test = train_models(X, y)
        results = evaluate_models(models, X_test_scaled, y_test)
        
        st.subheader('Model Performance')
        for model_name, result in results.items():
            st.write(f"\n{model_name}:")
            for name, m, r in zip(y.columns, result['mse'], result['r2']): 
                st.write(f'  {name}:')
                st.write(f'    Mean Squared Error: {m:.4f}')
                st.write(f'    R-squared Score: {r:.4f}')
        
        st.subheader('Actual vs Predicted Values')
        fig1 = plot_results(y_test, results, y.columns)
        st.pyplot(fig1)
        
        st.subheader('Model Comparison (Closing Price)')
        fig2 = plot_model_comparison(results, y.columns)
        st.pyplot(fig2)

if __name__ == '__main__':
    main()