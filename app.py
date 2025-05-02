import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import optimize
from datetime import datetime, timedelta
import base64

# App configuration
st.set_page_config(page_title="Bond Portfolio Optimizer", page_icon="📈", layout="wide")
st.title("Bond Portfolio Optimizer")
st.markdown("Optimize your bond portfolio based on risk, return, and liquidity preferences with market regime analysis.")

# Functions
def load_data(filename):
    """Load data from CSV file with error handling"""
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return pd.DataFrame()

def optimize_bond_portfolio(bond_data, forecast_df, user_inputs):
    """Optimizes bond portfolio based on user parameters using bond data and forecast data"""
    try:
        # Create a copy of bond data for optimization
        df = bond_data.copy()
        
        # Check for required columns in bond data
        required_columns = ['TICKER', 'MID_PRICE', 'MID_YIELD', 'DURATION_MID', 
                           'CONVEXITY_MID', 'BID_ASK_SPREAD', 'SPREAD_ROLLING_STD', 'VOLUME_ROLLING_STD']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {"success": False, "error": f"Missing columns in bond data: {missing_columns}"}
        
        # Find the date column in forecast data (allow for different naming conventions)
        if not forecast_df.empty:
            date_columns = [col for col in forecast_df.columns if col.lower() in ['date']]
            if not date_columns:
                return {"success": False, "error": "Missing date column in forecast data"}
            
            date_column = date_columns[0]
                
            # Ensure date column is datetime
            forecast_df[date_column] = pd.to_datetime(forecast_df[date_column])
                
            # Convert liquidity periods to datetime
            liquidity_periods = [[pd.to_datetime(start), pd.to_datetime(end)] 
                                for start, end in user_inputs['liquidity_periods']]
                
            # Determine ticker column in forecast data
            ticker_columns = [col for col in forecast_df.columns if col.lower() in ['ticker']]
            if not ticker_columns:
                return {"success": False, "error": "Missing ticker column in forecast data"}
            
            ticker_column = ticker_columns[0]
            
            # Find forecast column (try common names, then first numeric column)
            forecast_columns = [col for col in forecast_df.columns if col.lower() in ['forecast', 'predicted_liquidity']]
            
            if forecast_columns:
                forecast_column = forecast_columns[0]
            else:
                # Find first numeric column as fallback
                numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    forecast_column = numeric_cols[0]
                else:
                    return {"success": False, "error": "Missing forecast column"}
            
            # Convert ticker column to string type in forecast data
            forecast_df[ticker_column] = forecast_df[ticker_column].astype(str)
                
            # Calculate average liquidity for each ticker during specified periods
            ticker_liquidity = {}
            for start, end in liquidity_periods:
                date_mask = (forecast_df[date_column] >= start) & (forecast_df[date_column] <= end)
                period_data = forecast_df[date_mask]
                
                if not period_data.empty:
                    period_avg = period_data.groupby(ticker_column)[forecast_column].mean()
                    
                    for ticker, avg in period_avg.items():
                        if ticker in ticker_liquidity:
                            ticker_liquidity[ticker].append(avg)
                        else:
                            ticker_liquidity[ticker] = [avg]
            
            # Calculate overall average liquidity
            ticker_liquidity = {t: np.mean(vals) for t, vals in ticker_liquidity.items() if vals}
                
            # Create liquidity DataFrame and merge with bond data
            liquidity_df = pd.DataFrame({
                'ticker': list(ticker_liquidity.keys()),
                'predicted_liquidity': list(ticker_liquidity.values())
            })
            
            # Normalize ticker columns for joining
            ticker_col_in_df = next((col for col in df.columns if col.upper() == 'TICKER'), None)
            if not ticker_col_in_df:
                return {"success": False, "error": "Missing TICKER column in bond data"}
            
            # Convert ticker columns to string in both dataframes to ensure they can be merged properly
            df[ticker_col_in_df] = df[ticker_col_in_df].astype(str)
            liquidity_df['ticker'] = liquidity_df['ticker'].astype(str)
            
            df = pd.merge(df, liquidity_df, left_on=ticker_col_in_df, right_on='ticker', how='left')
            
            # Use the current liquidity_score for any missing predicted_liquidity values
            if 'liquidity_score' in df.columns:
                df['predicted_liquidity'] = df['predicted_liquidity'].fillna(df['liquidity_score'])
            else:
                # Create liquidity_score column if missing
                df['liquidity_score'] = df['predicted_liquidity'].fillna(1.0)
        else:
            # If forecast data is not available, use current liquidity score
            if 'liquidity_score' not in df.columns:
                df['liquidity_score'] = 1.0  # Default value
            df['predicted_liquidity'] = df['liquidity_score']
        
        # Filter for liquidity periods if we have date information in bond data
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            
            # Apply date filters based on liquidity periods
            date_masks = []
            for start, end in user_inputs['liquidity_periods']:
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)
                date_masks.append((df['DATE'] >= start_dt) & (df['DATE'] <= end_dt))
            
            if date_masks:
                combined_mask = date_masks[0]
                for mask in date_masks[1:]:
                    combined_mask = combined_mask | mask
                df = df[combined_mask]
        
        # Get latest data per ticker or use the provided averaged data
        if len(df) > len(df['TICKER'].unique()):
            # If there are multiple rows per ticker, get the latest data
            df = df.sort_values('DATE', ascending=False).groupby('TICKER').first().reset_index()
        
        # For deterministic results, seed with a hash of the preference weights
        seed_value = int(
            (user_inputs['return_weight'] * 13 + 
             user_inputs['risk_weight'] * 17 + 
             user_inputs['liquidity_weight'] * 19) % 10000
        )
        np.random.seed(seed_value)
        
        # Create covariance matrix based on bond characteristics and volatility data
        n = len(df)
        cov_matrix = np.zeros((n, n))
        
        # CHANGED: Create correlation scale with exponential scaling
        # Higher risk tolerance (higher risk_weight) = lower correlation damping
        def calculate_correlation_scale(risk_weight, market_regime="Normal"):
        # Base correlation parameters
            base_correlation = 0.15
            max_correlation = 0.60
            
            # IMPROVED: Correlation now increases with risk_weight
            risk_factor = (1 - np.exp(-0.3 * risk_weight)) / (1 + np.exp(-0.3 * risk_weight))
            
            # Scale between base and max correlation
            correlation_scale = base_correlation + (max_correlation - base_correlation) * risk_factor
            
            # Apply regime-specific adjustments
            regime_factors = {
                "High Volatility": 1.5,  # Correlations increase during stress
                "Normal": 1.0,           # No adjustment for normal regime
                "Low Volatility": 0.7    # Correlations decrease in calm markets
            }
            
            # Apply regime adjustment
            regime_factor = regime_factors.get(market_regime, 1.0)
            final_correlation = min(max_correlation, correlation_scale * regime_factor)
            
            return final_correlation
        correlation_scale = calculate_correlation_scale((user_inputs['risk_weight'] - 1), market_regime="Normal")
        
        # Extract bond characteristics for covariance calculation
        durations = df['DURATION_MID'].values
        yields = df['MID_YIELD'].values
        
        # Use the provided volatility metrics for improved risk estimation
        spread_vols = df['SPREAD_ROLLING_STD'].values
        volume_vols = df['VOLUME_ROLLING_STD'].values
        
        # Construct covariance matrix using both duration-based correlation and actual volatility data
        for i in range(n):
            for j in range(i, n):
                # Correlation based on duration similarity
                dur_diff = abs(durations[i] - durations[j])
                max_dur = max(durations) if durations.any() else 1.0
                dur_sim = 1 - dur_diff / max_dur
                
                # Incorporate spread and volume volatility data for more accurate covariance
                spread_vol_factor = (spread_vols[i] * spread_vols[j]) ** 0.5
                volume_vol_factor = (volume_vols[i] * volume_vols[j]) ** 0.5 if volume_vols.any() else 1.0
                
                # CHANGED: Adjust volatility weight with exponential scaling
                # At high risk levels, rely more on yields than historical volatility
                vol_weight = 0.8 * (0.9 ** (user_inputs['risk_weight'] - 1))
                cor_blend = dur_sim * (1 - vol_weight) + (1 / (1 + spread_vol_factor)) * vol_weight
                
                # Create covariance value 
                cov_val = cor_blend * yields[i] * yields[j] * 0.01 * correlation_scale * volume_vol_factor
                cov_matrix[i, j] = cov_val
                cov_matrix[j, i] = cov_val  # Symmetric matrix
                
        # Diagonal elements (variance) - use actual volatility data
        for i in range(n):
            # Blend historical vol with yield-based estimate for robustness
            yield_variance = (yields[i] * 0.03) ** 2
            historical_variance = (spread_vols[i]) ** 2 * (1 + volume_vols[i])
            
            # CHANGED: Adjust volatility weight with exponential scaling
            # At high risk levels, rely more on yields than historical volatility
            vol_weight = 0.8 * (0.9 ** (user_inputs['risk_weight'] - 1))
            blended_variance = yield_variance * (1 - vol_weight) + historical_variance * vol_weight
            
            # Set diagonal elements
            cov_matrix[i, i] = blended_variance * (1 + correlation_scale)
        
        # CHANGED: Normalize preference weights - but handle risk weight separately
        weights_sum = user_inputs['return_weight'] + user_inputs['liquidity_weight'] 
        norm_weights = {
            'return': user_inputs['return_weight'] / weights_sum,
            'liquidity': user_inputs['liquidity_weight'] / weights_sum,
        }
        
        # CHANGED: Create risk aversion parameter with exponential scaling
        # This maps risk slider (1-10) to risk aversion that works in opposite direction
        # Higher risk_weight = lower risk_aversion
        risk_aversion = 2.0 * (0.7 ** (user_inputs['risk_weight'] - 1))
        
        # Objective function for optimization
        def objective(weights):
            weights_1d = weights.flatten() if weights.ndim > 1 else weights
            
            # Return component (yield) - we want to maximize this
            portfolio_return = np.sum(weights_1d * df['MID_YIELD'].values)
            return_utility = portfolio_return * norm_weights['return'] * 3.0
            
            # CHANGED: Risk component with direct risk_aversion parameter
            portfolio_risk = np.sqrt(weights_1d.T @ cov_matrix @ weights_1d)
            risk_penalty = portfolio_risk * risk_aversion
            
            # Liquidity component - we want to maximize this
            portfolio_liquidity = np.sum(weights_1d * df['liquidity_score'].values)
            liquidity_utility = portfolio_liquidity * norm_weights['liquidity'] * 1.5
            
            # Bid-Ask Spread penalty - we want to minimize this
            spread_penalty = np.sum(weights_1d * df['BID_ASK_SPREAD'].values) * norm_weights['liquidity'] * 0.5
            
            # CHANGED: Diversification component with exponential scaling
            # Higher risk tolerance = less diversification required
            diversification_exponent = 1.0 + (1.0 * (0.9 ** (user_inputs['risk_weight'] - 1)))
            concentration = np.sum(weights_1d ** diversification_exponent)
            diversification_penalty = concentration * (0.5 * (0.8 ** (user_inputs['risk_weight'] - 1)))
            
            # Combined utility function (negative because we're minimizing)
            utility = (return_utility - risk_penalty + liquidity_utility - spread_penalty - diversification_penalty)
            
            return -utility
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Weights sum to 1
        
        # CHANGED: Maximum allowed weight with exponential scaling
        # Maps risk_weight from 1-10 to max_weight from ~20% to ~50%
        max_weight = 0.2 + 0.3 * (1 - (0.85 ** (user_inputs['risk_weight'] - 1)))
        max_weight = min(max_weight, 0.5)  # Cap at 50%
        
        bounds = tuple((0, max_weight) for _ in range(len(df)))
        
        # Initial weights for optimization
        equal_weights = np.ones(len(df)) / len(df)
        
        # Optimization with multiple starting points
        results = []
        
        # Start with equal weights
        result = optimize.minimize(
            objective, equal_weights, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result['success']:
            results.append(result)
        
        # Try more random starting points for better convergence
        for _ in range(3):
            rand_weights = np.random.rand(len(df))
            rand_weights = rand_weights / np.sum(rand_weights)
            
            result = optimize.minimize(
                objective, rand_weights, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result['success']:
                results.append(result)
        
        if not results:
            return {"success": False, "error": "Optimization failed to converge"}
            
        # Choose the best result
        best_result = min(results, key=lambda x: x['fun'])
        
        # Final weights and portfolio metrics
        weights = best_result['x']
        
        # Calculate portfolio metrics
        port_return = np.sum(weights * df['MID_YIELD'].values)
        port_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        port_liquidity = np.sum(weights * df['liquidity_score'].values)
        sharpe_ratio = port_return / port_risk if port_risk > 0 else 0
        
        # Calculate allocations
        prices = df['MID_PRICE'].values
        allocations = weights * user_inputs['budget']
        quantities = np.floor(allocations / prices)
        actual_allocation = quantities * prices
        
        # Get significant positions (>0.5%)
        significant_indices = np.where(weights > 0.005)[0]
        
        # Create summary DataFrame
        portfolio_summary = pd.DataFrame({
            'ticker': df.iloc[significant_indices]['TICKER'].values,
            'weight': weights[significant_indices],
            'quantity': quantities[significant_indices],
            'allocation': actual_allocation[significant_indices],
            'yield': df.iloc[significant_indices]['MID_YIELD'].values,
            'duration': df.iloc[significant_indices]['DURATION_MID'].values,
            'liquidity_score': df.iloc[significant_indices]['liquidity_score'].values,
            'bid_ask_spread': df.iloc[significant_indices]['BID_ASK_SPREAD'].values,
        })
        
        # Calculate diversity metrics
        herfindahl_index = np.sum(portfolio_summary['weight']**2)
        
        # Remove duplicates and recalculate weights
        portfolio_summary = portfolio_summary.drop_duplicates(subset=['ticker'])
        
        # Check if portfolio is empty after filtering
        if len(portfolio_summary) > 0:
            # Normalize weights to sum to 1 after filtering
            portfolio_summary['weight'] = portfolio_summary['weight'] / portfolio_summary['weight'].sum()
            portfolio_summary['allocation'] = portfolio_summary['weight'] * user_inputs['budget']
            
            # Recalculate quantities based on new allocations
            portfolio_summary['quantity'] = np.floor(portfolio_summary['allocation'] / 
                                           portfolio_summary['ticker'].map(
                                               dict(zip(df['TICKER'], df['MID_PRICE']))))
        
        # Reset index
        portfolio_summary = portfolio_summary.reset_index(drop=True)
        
        return {
            "success": True,
            'portfolio_summary': portfolio_summary,
            'metrics': {
                'expected_return': port_return,
                'risk': port_risk,
                'liquidity': port_liquidity,
                'sharpe_ratio': sharpe_ratio,
                'diversification': 1 - herfindahl_index
            },
            'bond_data': df,  # Include for visualization
            'weights': weights,      # Raw weights for stress testing
            'cov_matrix': cov_matrix # Covariance matrix for stress testing
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def stress_test_portfolio(portfolio_results, bond_data, selected_regime):
    """
    Perform stress testing on the optimized portfolio under different market regimes
    
    Parameters:
    -----------
    portfolio_results: Dict with portfolio optimization results
    bond_data: DataFrame with bond data including market regime information
    selected_regime: String indicating which regime to test against
    
    Returns:
    --------
    Dict with stress test results
    """
    try:
        if not portfolio_results["success"]:
            return {"success": False, "error": "No valid portfolio to stress test"}
        
        # Get regime-specific data
        regime_subset = bond_data[bond_data['MARKET_REGIME'] == selected_regime]
        
        if len(regime_subset) == 0:
            # If no data for the selected regime, try to infer regime characteristics
            if 'vol_regime' in bond_data.columns:
                # See if vol_regime might contain the selected regime
                regime_subset = bond_data[bond_data['vol_regime'] == selected_regime]
            
            if len(regime_subset) == 0:
                return {"success": False, "error": f"No data available for regime: {selected_regime}"}
        
        # Calculate average regime characteristics
        avg_volatility = regime_subset['SPREAD_ROLLING_STD'].mean() if 'SPREAD_ROLLING_STD' in regime_subset.columns else 0.015
        avg_volume_vol = regime_subset['VOLUME_ROLLING_STD'].mean() if 'VOLUME_ROLLING_STD' in regime_subset.columns else 0.5
        avg_yield_spread = regime_subset['YIELD_SPREAD'].mean() if 'YIELD_SPREAD' in regime_subset.columns else 0.5
        
        # Get portfolio data
        portfolio = portfolio_results['portfolio_summary']
        weights = portfolio_results['weights']
        bond_data = portfolio_results['bond_data']
        cov_matrix = portfolio_results['cov_matrix']
        
        # Stress factors based on regime
        stress_factors = {
            'High Volatility': {
                'yield_shift': 0.5,     # 50 bps increase in yields
                'vol_multiplier': 2.0,  # Double volatility
                'liquidity_factor': 0.6, # 40% reduction in liquidity
                'correlation_increase': 0.3 # Increase in correlations
            },
            'Normal': {
                'yield_shift': 0.0,     # No change in yields
                'vol_multiplier': 1.0,  # No change in volatility
                'liquidity_factor': 1.0, # No change in liquidity
                'correlation_increase': 0.0 # No change in correlations
            },
            'Low Volatility': {
                'yield_shift': -0.2,    # 20 bps decrease in yields
                'vol_multiplier': 0.7,  # 30% reduction in volatility
                'liquidity_factor': 1.2, # 20% increase in liquidity
                'correlation_increase': -0.1 # Decrease in correlations
            }
        }
        
        # Use default if regime not found
        if selected_regime not in stress_factors:
            # Try to infer if it's high, normal or low volatility based on name
            if 'high' in selected_regime.lower():
                stress_factors[selected_regime] = stress_factors['High Volatility']
            elif 'low' in selected_regime.lower():
                stress_factors[selected_regime] = stress_factors['Low Volatility']
            else:
                stress_factors[selected_regime] = stress_factors['Normal']
        
        # Apply stress factors
        factors = stress_factors[selected_regime]
        
        # Adjusted yields - apply yield shift
        stressed_yields = bond_data['MID_YIELD'].values + factors['yield_shift']
        
        # Adjusted prices (inverse relationship with yields, simplified)
        durations = bond_data['DURATION_MID'].values
        price_impact = -durations * factors['yield_shift'] / 100
        stressed_prices = bond_data['MID_PRICE'].values * (1 + price_impact)
        
        # Adjusted covariance matrix
        stressed_cov = cov_matrix * factors['vol_multiplier']
        
        # Increase correlations during stress
        if factors['correlation_increase'] != 0:
            for i in range(len(stressed_cov)):
                for j in range(len(stressed_cov)):
                    if i != j:
                        # Increase off-diagonal elements (correlations)
                        current_cov = stressed_cov[i, j]
                        max_possible_cov = np.sqrt(stressed_cov[i, i] * stressed_cov[j, j])
                        # Ensure we don't exceed maximum possible correlation (1.0)
                        new_cov = min(current_cov * (1 + factors['correlation_increase']), 
                                     max_possible_cov * 0.999)
                        stressed_cov[i, j] = new_cov
                        stressed_cov[j, i] = new_cov  # Keep matrix symmetric
        
        # Calculate stressed portfolio metrics
        stressed_return = np.sum(weights * stressed_yields)
        stressed_risk = np.sqrt(weights.T @ stressed_cov @ weights)
        
        # Adjust liquidity based on regime
        liquidity_adjustment = factors['liquidity_factor']
        stressed_liquidity = portfolio_results['metrics']['liquidity'] * liquidity_adjustment
        
        # Calculate price impact
        price_return = np.sum(weights * price_impact) * 100  # Convert to percentage
        
        # Calculate stress test metrics
        stress_results = {
            'original_metrics': portfolio_results['metrics'],
            'stressed_metrics': {
                'expected_return': stressed_return,
                'risk': stressed_risk,
                'liquidity': stressed_liquidity,
                'sharpe_ratio': stressed_return / stressed_risk if stressed_risk > 0 else 0,
                'price_impact': price_return
            },
            'regime_characteristics': {
                'avg_volatility': avg_volatility,
                'avg_volume_vol': avg_volume_vol,
                'avg_yield_spread': avg_yield_spread
            }
        }
        
        return {"success": True, "results": stress_results}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# Load data - both bond data and forecast data
bond_data = load_data("bond_data.csv")
forecast_df = load_data("forecast_data.csv")

# Add this cleaning code after loading bond data
if not bond_data.empty:
    # Convert Date column to datetime if it exists
    date_cols = [col for col in bond_data.columns if col.lower() == 'date']
    if date_cols:
        bond_data[date_cols[0]] = pd.to_datetime(bond_data[date_cols[0]])
    
    # Find and clean regime name column
    regime_cols = [col for col in bond_data.columns if col.upper() in ['MARKET_REGIME', 'VOL_REGIME']]
    for regime_col in regime_cols:
        # Convert all regime names to strings to ensure consistency
        bond_data[regime_col] = bond_data[regime_col].astype(str)

# Clean forecast data
if not forecast_df.empty:
    # Convert Date column to datetime if it exists
    date_cols = [col for col in forecast_df.columns if col.lower() == 'date']
    if date_cols:
        forecast_df[date_cols[0]] = pd.to_datetime(forecast_df[date_cols[0]])
        
        # Optional: Clean up regime names - remove any that are just numeric/empty
        # This depends on your specific use case
        # bond_data = bond_data[bond_data[regime_col].str.strip() != '']
        # bond_data = bond_data[~bond_data[regime_col].str.isnumeric()]

# Sidebar inputs
st.sidebar.header("Optimization Parameters")
budget = st.sidebar.number_input("Investment Budget ($)", min_value=10000, max_value=10000000, value=1000000, step=100000)

st.sidebar.subheader("Liquidity Period")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() + timedelta(days=30))
end_date = st.sidebar.date_input("End Date", value=datetime.now() + timedelta(days=90))

st.sidebar.subheader("Investment Preferences")
return_weight = st.sidebar.slider("Return Weight", min_value=1, max_value=10, value=5, 
                               help="Higher values prioritize yield maximization")
# CHANGED: Updated help text for risk_weight
risk_weight = st.sidebar.slider("Risk Weight", min_value=1, max_value=10, value=5, 
                             help="Higher values allow more risk-taking for potentially higher returns")
liquidity_weight = st.sidebar.slider("Liquidity Weight", min_value=1, max_value=10, value=5, 
                                  help="Higher values prioritize better liquidity")

# User inputs for optimization
user_inputs = {
    'budget': budget,
    'liquidity_periods': [[start_date, end_date]],
    'return_weight': return_weight,
    'risk_weight': risk_weight,
    'liquidity_weight': liquidity_weight
}

# Initialize session state variables
if 'optimization_results' not in st.session_state:
    st.session_state['optimization_results'] = None

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Portfolio Optimizer", "Stress Testing", "Bond Analysis"])

with tab1:
    st.write("Configure your investment preferences in the sidebar and click 'Run Optimization' to generate an optimized bond portfolio.")
    
    if st.button("Run Optimization", type="primary"):
        if bond_data.empty:
            st.error("Cannot run optimization without bond data. Please ensure data files are available.")
        else:
            with st.spinner("Optimizing portfolio..."):
                results = optimize_bond_portfolio(bond_data, forecast_df, user_inputs)
                
                # Store results in session state
                st.session_state['optimization_results'] = results
                
                if results["success"]:
                    st.success("Portfolio optimization complete!")
                    
                    # Display metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Expected Return", f"{results['metrics']['expected_return']:.2f}%")
                    
                    with col2:
                        st.metric("Portfolio Risk", f"{results['metrics']['risk']:.2f}%")
                    
                    with col3:
                        st.metric("Liquidity Score", f"{results['metrics']['liquidity']:.2f}")
                    
                    with col4:
                        st.metric("Sharpe Ratio", f"{results['metrics']['sharpe_ratio']:.2f}")
                    
                    # Display portfolio allocation
                    st.subheader("Portfolio Allocation")
                    
                    # Format the portfolio summary for display
                    display_df = results['portfolio_summary'].copy()
                    if not display_df.empty:
                        display_df['weight'] = display_df['weight'].apply(lambda x: f"{x:.2%}")
                        display_df['yield'] = display_df['yield'].apply(lambda x: f"{x:.2f}%")
                        display_df['allocation'] = display_df['allocation'].apply(lambda x: f"${x:,.2f}")
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Interactive Pie Chart with Plotly
                        st.subheader("Portfolio Composition")
                        
                        portfolio = results['portfolio_summary']
                        
                        # Only show top 10 holdings for clarity if there are many
                        if len(portfolio) > 10:
                            plot_data = portfolio.sort_values('weight', ascending=False).head(10).copy()
                            other_weight = portfolio.iloc[10:]['weight'].sum()
                            # Only add 'Other' if there are actually more holdings
                            if other_weight > 0:
                                other_row = pd.DataFrame({
                                    'ticker': ['Other'],
                                    'weight': [other_weight],
                                    'allocation': [other_weight * user_inputs['budget']]
                                })
                                plot_data = pd.concat([plot_data, other_row])
                        else:
                            plot_data = portfolio.copy()
                        
                        # Hover info formatting
                        plot_data['hover_text'] = plot_data.apply(
                            lambda row: f"Ticker: {row['ticker']}<br>" +
                                       f"Weight: {row['weight']:.2%}<br>" +
                                       f"Allocation: ${row['allocation']:,.2f}",
                            axis=1
                        )
                        
                        # Plotly pie chart
                        fig = px.pie(
                            plot_data, 
                            values='weight', 
                            names='ticker',
                            title='Portfolio Allocation by Weight',
                            hover_data=['allocation'],
                            custom_data=['hover_text']
                        )
                        
                        fig.update_traces(
                            hovertemplate='%{customdata[0]}<extra></extra>',
                            textinfo='label+percent'
                        )
                        
                        fig.update_layout(
                            height=500,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.3,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interactive Risk/Return Scatter Plot for individual bonds
                        fig2 = px.scatter(
                            results['bond_data'],
                            x='DURATION_MID',
                            y='MID_YIELD',
                            color='liquidity_score',
                            size='MID_PRICE',
                            hover_name='TICKER',
                            color_continuous_scale=px.colors.sequential.Viridis,
                            title='Bond Characteristics: Yield vs Duration',
                            labels={
                                'DURATION_MID': 'Duration (Years)',
                                'MID_YIELD': 'Yield (%)',
                                'liquidity_score': 'Liquidity Score'
                            }
                        )
                        
                        # Highlight selected bonds
                        selected_bonds = results['portfolio_summary']['ticker'].tolist()
                        highlight_df = results['bond_data'][results['bond_data']['TICKER'].isin(selected_bonds)]
                        
                        if not highlight_df.empty:
                            fig2.add_trace(
                                go.Scatter(
                                    x=highlight_df['DURATION_MID'],
                                    y=highlight_df['MID_YIELD'],
                                    mode='markers',
                                    marker=dict(
                                        size=15,
                                        symbol='circle-open',
                                        line=dict(width=3, color='red'),
                                        opacity=1
                                    ),
                                    name='Selected Bonds',
                                    hovertext=highlight_df['TICKER']
                                )
                            )
                            
                        fig2.update_layout(
                            height=500,
                            xaxis_title="Duration (Years)",
                            yaxis_title="Yield (%)",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.3,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Add download link
                        csv = results['portfolio_summary'].to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="portfolio_allocation.csv">Download Portfolio Allocation CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    else:
                        st.warning("No bonds with significant weights were found in the optimized portfolio.")
                else:
                    st.error(f"Optimization failed: {results.get('error', 'Unknown error')}")
    else:
        st.info("Click 'Run Optimization' to generate an optimized bond portfolio based on your preferences.")

with tab2:
    st.subheader("Portfolio Stress Testing")
    
    if st.session_state.get('optimization_results') and st.session_state['optimization_results'].get('success'):
        # Get available regimes from market regime column
        regimes = []
        
        if not bond_data.empty:
            # Check for market regime column first
            if 'MARKET_REGIME' in bond_data.columns:
                regimes = sorted(bond_data['MARKET_REGIME'].unique())
            # Then check for vol_regime as backup
            elif 'vol_regime' in bond_data.columns:
                regimes = sorted(bond_data['vol_regime'].unique())
            
            if regimes:
                selected_regime = st.selectbox("Select Market Regime for Stress Test", regimes)
                
                if st.button("Run Stress Test"):
                    with st.spinner("Running stress test..."):
                        stress_results = stress_test_portfolio(
                            st.session_state['optimization_results'], 
                            bond_data, 
                            selected_regime
                        )
                        
                        if stress_results["success"]:
                            results = stress_results["results"]
                            
                            # Display comparison of original vs stressed metrics
                            st.subheader(f"Stress Test Results: {selected_regime} Regime")
                            
                            # Create comparison metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            # Return
                            with col1:
                                original = results['original_metrics']['expected_return']
                                stressed = results['stressed_metrics']['expected_return']
                                delta = stressed - original
                                st.metric("Expected Return", 
                                         f"{stressed:.2f}%",
                                         f"{delta:.2f}%",
                                         delta_color="normal")
                            
                            # Risk
                            with col2:
                                original = results['original_metrics']['risk']
                                stressed = results['stressed_metrics']['risk']
                                delta = stressed - original
                                st.metric("Portfolio Risk", 
                                         f"{stressed:.2f}%",
                                         f"{delta:.2f}%",
                                         delta_color="inverse")
                            
                            # Liquidity
                            with col3:
                                original = results['original_metrics']['liquidity']
                                stressed = results['stressed_metrics']['liquidity']
                                delta = stressed - original
                                st.metric("Liquidity Score", 
                                         f"{stressed:.2f}",
                                         f"{delta:.2f}",
                                         delta_color="normal")
                            
                            # Sharpe Ratio
                            with col4:
                                original = results['original_metrics']['sharpe_ratio']
                                stressed = results['stressed_metrics']['sharpe_ratio']
                                delta = stressed - original
                                st.metric("Sharpe Ratio", 
                                         f"{stressed:.2f}",
                                         f"{delta:.2f}",
                                         delta_color="normal")
                            
                            # Display price impact
                            st.metric("Portfolio Price Impact", 
                                    f"{results['stressed_metrics']['price_impact']:.2f}%")
                            
                            # Characteristics of the selected regime
                            st.subheader("Regime Characteristics")
                            st.write(f"Average Spread Volatility: {results['regime_characteristics']['avg_volatility']:.4f}")
                            st.write(f"Average Volume Volatility: {results['regime_characteristics']['avg_volume_vol']:.4f}")
                            st.write(f"Average Yield Spread: {results['regime_characteristics']['avg_yield_spread']:.4f}")
                            
                            # Visual comparison
                            comparison_data = pd.DataFrame({
                                'Metric': ['Expected Return', 'Risk', 'Liquidity', 'Sharpe Ratio'],
                                'Original': [
                                    results['original_metrics']['expected_return'],
                                    results['original_metrics']['risk'],
                                    results['original_metrics']['liquidity'],
                                    results['original_metrics']['sharpe_ratio']
                                ],
                                'Stressed': [
                                    results['stressed_metrics']['expected_return'],
                                    results['stressed_metrics']['risk'],
                                    results['stressed_metrics']['liquidity'],
                                    results['stressed_metrics']['sharpe_ratio']
                                ]
                            })
                            
                            # Reshape for plotting
                            plot_data = pd.melt(comparison_data, 
                                              id_vars=['Metric'], 
                                              value_vars=['Original', 'Stressed'],
                                              var_name='Scenario', value_name='Value')
                            
                            # Create comparison chart
                            fig = px.bar(plot_data, x='Metric', y='Value', color='Scenario',
                                       barmode='group',
                                       title=f'Portfolio Metrics: Original vs {selected_regime} Regime',
                                       labels={'Value': 'Metric Value', 'Metric': 'Metric'},
                                       color_discrete_map={'Original': 'blue', 'Stressed': 'orange'})
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error(f"Stress test failed: {stress_results.get('error', 'Unknown error')}")
            else:
                st.error("No regime information found in the bond data.")
        else:
            st.error("Bond data is not available for stress testing.")
    else:
        st.info("Please run a portfolio optimization first in the 'Portfolio Optimizer' tab.")

with tab3:
    st.subheader("Bond Analysis Dashboard")
    
    if not bond_data.empty:
        # Filter for analysis
        st.write("Analyze the bond universe to understand available options.")
        
        # Get duration range
        if 'DURATION_MID' in bond_data.columns:
            min_dur = float(bond_data['DURATION_MID'].min())
            max_dur = float(bond_data['DURATION_MID'].max())
            
            # Duration filter
            duration_range = st.slider("Duration Range (Years)", 
                                     min_value=min_dur, 
                                     max_value=max_dur,
                                     value=(min_dur, max_dur))
            
            # Filter bonds by duration
            filtered_df = bond_data[(bond_data['DURATION_MID'] >= duration_range[0]) & 
                                  (bond_data['DURATION_MID'] <= duration_range[1])]
            
            # Check if yield column exists
            if 'MID_YIELD' in filtered_df.columns:
                # Create yield curve visualization
                st.subheader("Yield Curve Analysis")
                
                # Group by duration (rounded) and get average yield
                filtered_df['Duration_Rounded'] = filtered_df['DURATION_MID'].round(1)
                yield_curve = filtered_df.groupby('Duration_Rounded')['MID_YIELD'].mean().reset_index()
                
                # Plot yield curve
                fig = px.line(yield_curve, 
                             x='Duration_Rounded', 
                             y='MID_YIELD',
                             markers=True,
                             title='Current Yield Curve',
                             labels={
                                 'Duration_Rounded': 'Duration (Years)',
                                 'MID_YIELD': 'Yield (%)'
                             })
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Volatility analysis if spread_rolling_std exists
                if 'SPREAD_ROLLING_STD' in filtered_df.columns:
                    st.subheader("Volatility Analysis")
                    
                    # Create volatility vs yield scatter plot
                    fig_vol = px.scatter(
                        filtered_df,
                        x='SPREAD_ROLLING_STD',
                        y='MID_YIELD',
                        color='DURATION_MID',
                        size='VOLUME_ROLLING_STD' if 'VOLUME_ROLLING_STD' in filtered_df.columns else None,
                        hover_name='TICKER',
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title='Risk vs Return: Yield vs Volatility',
                        labels={
                            'SPREAD_ROLLING_STD': 'Spread Volatility',
                            'MID_YIELD': 'Yield (%)',
                            'DURATION_MID': 'Duration (Years)',
                            'VOLUME_ROLLING_STD': 'Volume Volatility'
                        }
                    )
                    
                    fig_vol.update_layout(height=400)
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                # Market Regime Analysis
                if 'MARKET_REGIME' in filtered_df.columns:
                    st.subheader("Market Regime Analysis")
                    
                    # Group by regime and get average metrics
                    regime_metrics = filtered_df.groupby('MARKET_REGIME').agg({
                        'MID_YIELD': 'mean',
                        'SPREAD_ROLLING_STD': 'mean' if 'SPREAD_ROLLING_STD' in filtered_df.columns else 'count',
                        'VOLUME_ROLLING_STD': 'mean' if 'VOLUME_ROLLING_STD' in filtered_df.columns else 'count',
                        'YIELD_SPREAD': 'mean' if 'YIELD_SPREAD' in filtered_df.columns else 'count',
                        'BID_ASK_SPREAD': 'mean'
                    }).reset_index()
                    
                    # Plot regime comparison
                    fig_regime = px.bar(
                        regime_metrics,
                        x='MARKET_REGIME',
                        y='MID_YIELD',
                        color='MARKET_REGIME',
                        title='Average Yield by Market Regime',
                        labels={
                            'MARKET_REGIME': 'Market Regime',
                            'MID_YIELD': 'Average Yield (%)'
                        }
                    )
                    
                    fig_regime.update_layout(height=400)
                    st.plotly_chart(fig_regime, use_container_width=True)
                
                # Scatter plot of all bonds
                st.subheader("Bond Universe")
                
                if 'liquidity_score' in filtered_df.columns:
                    fig2 = px.scatter(
                        filtered_df,
                        x='DURATION_MID',
                        y='MID_YIELD',
                        color='liquidity_score',
                        size='MID_PRICE',
                        hover_name='TICKER',
                        color_continuous_scale=px.colors.sequential.Viridis,
                        title='Bond Universe: Yield vs Duration',
                        labels={
                            'DURATION_MID': 'Duration (Years)',
                            'MID_YIELD': 'Yield (%)',
                            'liquidity_score': 'Liquidity Score',
                            'MID_PRICE': 'Price'
                        }
                    )
                else:
                    fig2 = px.scatter(
                        filtered_df,
                        x='DURATION_MID',
                        y='MID_YIELD',
                        size='MID_PRICE',
                        hover_name='TICKER',
                        title='Bond Universe: Yield vs Duration',
                        labels={
                            'DURATION_MID': 'Duration (Years)',
                            'MID_YIELD': 'Yield (%)',
                            'MID_PRICE': 'Price'
                        }
                    )
                
                fig2.update_layout(height=500)
                st.plotly_chart(fig2, use_container_width=True)
                
                # Statistical summary
                st.subheader("Bond Universe Statistics")
                
                # Create summary statistics
                if len(filtered_df) > 0:
                    stats_df = pd.DataFrame({
                        'Metric': ['Count', 'Avg Duration', 'Avg Yield', 'Avg Price', 'Min Yield', 'Max Yield',
                                  'Avg Spread Volatility', 'Avg Volume Volatility', 'Avg Liquidity'],
                        'Value': [
                            len(filtered_df),
                            filtered_df['DURATION_MID'].mean(),
                            filtered_df['MID_YIELD'].mean(),
                            filtered_df['MID_PRICE'].mean(),
                            filtered_df['MID_YIELD'].min(),
                            filtered_df['MID_YIELD'].max(),
                            filtered_df['SPREAD_ROLLING_STD'].mean() if 'SPREAD_ROLLING_STD' in filtered_df.columns else 'N/A',
                            filtered_df['VOLUME_ROLLING_STD'].mean() if 'VOLUME_ROLLING_STD' in filtered_df.columns else 'N/A',
                            filtered_df['liquidity_score'].mean() if 'liquidity_score' in filtered_df.columns else 'N/A'
                        ]
                    })
                    
                    # Format values
                    stats_df['Value'] = stats_df['Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
                    
                    # Display as a table
                    st.table(stats_df)
                else:
                    st.warning("No bonds found with the selected criteria.")
            else:
                st.error("MID_YIELD column not found in the bond data.")
        else:
            st.error("DURATION_MID column not found in the bond data.")
    else:
        st.error("Bond data is not available for analysis.")