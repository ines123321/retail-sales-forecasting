from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import traceback
import os
from sklearn.preprocessing import OrdinalEncoder  # Ajout de l'import manquant
import matplotlib.pyplot as plt
import base64
# Ajoutez en haut du fichier avec les autres imports
import io
import matplotlib
matplotlib.use('Agg')  # Important pour éviter les problèmes de thread avec Flask
from matplotlib.dates import DateFormatter  # <-- Ajout crucial
import matplotlib.dates as mdates

app = Flask(__name__)

# Load historical data at startup
print("Loading historical data...")
historical_df = pd.read_csv('dataset_final_nettoye.csv', sep=';', parse_dates=['fk_jour'], dayfirst=True)
historical_df = historical_df.sort_values('fk_jour')

# Calculate global means for fallback values
global_qte_mean = historical_df['qte'].mean()
global_ticket_mean = historical_df['nbr_ticket'].mean()
global_ca_mean = historical_df['ca_ttc'].mean()

def get_historical_values(article_id, magasin_id, days_back=7):
    """Get historical values for an article/store combination"""
    mask = (historical_df['fk_article'] == article_id) & (historical_df['fk_magasin'] == magasin_id)
    article_data = historical_df[mask].copy()
    
    if len(article_data) == 0:
        return None
    
    # Take the last record per day
    article_data = article_data.groupby('fk_jour').agg({
        'qte': 'sum',
        'nbr_ticket': 'sum',
        'ca_ttc': 'sum'
    }).reset_index()
    
    # Get the most recent days
    article_data = article_data.sort_values('fk_jour', ascending=False)
    return article_data.head(days_back)

def calculate_lag_features(history, target_date):
    """Calculate lag features from historical data"""
    features = {}
    
    # Calculate article-specific means if history exists
    if history is not None and len(history) > 0:
        article_qte_mean = history['qte'].mean()
        article_ticket_mean = history['nbr_ticket'].mean()
        article_ca_mean = history['ca_ttc'].mean()
    else:
        article_qte_mean = global_qte_mean
        article_ticket_mean = global_ticket_mean
        article_ca_mean = global_ca_mean
    
    if history is None or len(history) < 7:
        # Use more realistic default values based on article means
        for lag in [1, 2, 7]:
            features[f'qte_lag_{lag}'] = article_qte_mean
            features[f'nbr_ticket_lag_{lag}'] = article_ticket_mean
            features[f'ca_lag_{lag}'] = article_ca_mean
            features[f'tickets_lag_{lag}'] = article_ticket_mean
        
        features['qte_rolling_mean_7'] = article_qte_mean
        features['nbr_ticket_rolling_mean_7'] = article_ticket_mean
        features['tickets_rolling_mean_7'] = article_ticket_mean
        features['ca_rolling_mean_7'] = article_ca_mean
        features['ca_rolling_max_7'] = article_ca_mean
        features['ratio_lag1_rolling7'] = 1.0
    else:
        # Calculate actual lag features
        history = history.sort_values('fk_jour')
        
        # Quantity features
        for lag in [1, 2, 7]:
            if len(history) >= lag:
                features[f'qte_lag_{lag}'] = history['qte'].iloc[-lag]
            else:
                features[f'qte_lag_{lag}'] = article_qte_mean
        
        # Ticket features
        for lag in [1, 7]:
            if len(history) >= lag:
                features[f'nbr_ticket_lag_{lag}'] = history['nbr_ticket'].iloc[-lag]
                features[f'tickets_lag_{lag}'] = history['nbr_ticket'].iloc[-lag]
            else:
                features[f'nbr_ticket_lag_{lag}'] = article_ticket_mean
                features[f'tickets_lag_{lag}'] = article_ticket_mean
        
        # Revenue features
        for lag in [1, 7]:
            if len(history) >= lag:
                features[f'ca_lag_{lag}'] = history['ca_ttc'].iloc[-lag]
            else:
                features[f'ca_lag_{lag}'] = article_ca_mean
        
        # Rolling features (7 days)
        if len(history) >= 7:
            features['qte_rolling_mean_7'] = history['qte'].tail(7).mean()
            features['nbr_ticket_rolling_mean_7'] = history['nbr_ticket'].tail(7).mean()
            features['tickets_rolling_mean_7'] = history['nbr_ticket'].tail(7).mean()
            features['ca_rolling_mean_7'] = history['ca_ttc'].tail(7).mean()
            features['ca_rolling_max_7'] = history['ca_ttc'].tail(7).max()
        else:
            features['qte_rolling_mean_7'] = article_qte_mean
            features['nbr_ticket_rolling_mean_7'] = article_ticket_mean
            features['tickets_rolling_mean_7'] = article_ticket_mean
            features['ca_rolling_mean_7'] = article_ca_mean
            features['ca_rolling_max_7'] = article_ca_mean
        
        # Ratio feature
        if len(history) >= 7 and features['qte_lag_1'] > 0 and features['qte_rolling_mean_7'] > 0:
            features['ratio_lag1_rolling7'] = features['qte_lag_1'] / features['qte_rolling_mean_7']
        else:
            features['ratio_lag1_rolling7'] = 1.0
    
    return features

def create_all_features(form_data):
    """Create all possible features that any model might need"""
    date_obj = datetime.strptime(form_data['date'], '%Y-%m-%d')
    article_id = int(form_data['article_id'])
    magasin_id = int(form_data['magasin_id'])
    
    # Get historical data for this article/store
    history = get_historical_values(article_id, magasin_id)
    lag_features = calculate_lag_features(history, date_obj)
    
    # Temporal features
    day_of_year = date_obj.timetuple().tm_yday
    month = date_obj.month
    day_of_week = date_obj.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    is_month_end = 1 if date_obj.day >= 28 else 0
    
    # Initialize France holidays
    fr_holidays = holidays.France()
    
    # Calculate article and store means from historical data
    article_data = historical_df[historical_df['fk_article'] == article_id]
    magasin_data = historical_df[historical_df['fk_magasin'] == magasin_id]
    
    article_mean = article_data['ca_ttc'].mean() if not article_data.empty else global_ca_mean
    magasin_mean = magasin_data['ca_ttc'].mean() if not magasin_data.empty else global_ca_mean
    article_qte_mean = article_data['qte'].mean() if not article_data.empty else global_qte_mean
    
    # Create ALL possible features
    features = {
        # Basic features
        'fk_article': article_id,
        'fk_magasin': magasin_id,
        'fk_article_encoded': article_id % 100,
        'fk_magasin_encoded': magasin_id % 100,
        
        # Temporal features
        'day_of_year': day_of_year,
        'month': month,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_month_end': is_month_end,
        
        # Cyclic features
        'day_of_year_sin': np.sin(2 * np.pi * day_of_year / 365),
        'day_of_year_cos': np.cos(2 * np.pi * day_of_year / 365),
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'jour_sin': np.sin(2 * np.pi * day_of_year / 365),
        'jour_cos': np.cos(2 * np.pi * day_of_year / 365),
        
        # Special days
        'is_holiday': 1 if date_obj.date() in fr_holidays else 0,
        'is_summer': 1 if month >= 6 and month <= 8 else 0,
        
        # Aggregated features
        'article_mean': article_mean,
        'magasin_mean': magasin_mean,
        'article_weekend': is_weekend * article_mean,
        'weekend_effect': is_weekend * article_mean,
        'lag1_weekend': is_weekend * article_mean,
    }
    
    # Add the lag features
    features.update(lag_features)
    
    return pd.DataFrame([features])


def generate_time_series_plot(article_id, model_type, model, start_date, days_to_predict=210):
    """Generate time series prediction plot that better matches historical variability"""
    try:
        # Prepare date range
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = start_date + timedelta(days=days_to_predict)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Get historical data
        history = historical_df[historical_df['fk_article'] == article_id]
        history = history.groupby('fk_jour').agg({
            'qte': 'sum',
            'nbr_ticket': 'sum',
            'ca_ttc': 'sum'
        }).reset_index().sort_values('fk_jour')
        
        # Analyze historical patterns
        target_col = model_type if model_type != 'ticket' else 'nbr_ticket'
        hist_stats = {}
        
        if len(history) > 0:
            # Calculate daily variability patterns
            history['day_of_week'] = history['fk_jour'].dt.dayofweek
            daily_stats = history.groupby('day_of_week')[target_col].agg(['mean', 'std', 'median', 'max'])
            
            # Calculate overall stats
            overall_mean = history[target_col].mean()
            overall_std = history[target_col].std()
            
            hist_stats = {
                'daily': daily_stats.to_dict(orient='index'),
                'overall_mean': overall_mean,
                'overall_std': overall_std,
                'max_value': history[target_col].max(),
                'min_value': history[target_col].min()
            }
        
        # Generate predictions with historical variability
        predictions = []
        recent_values = []
        
        for i, date in enumerate(dates):
            # Create base features
            form_data = {
                'article_id': str(article_id),
                'magasin_id': '0',
                'date': date.strftime('%Y-%m-%d'),
                'model_type': model_type
            }
            input_df = create_all_features(form_data)
            
            # Get day characteristics
            day_of_week = date.weekday()
            is_weekend = day_of_week >= 5
            is_holiday = date.date() in holidays.France()
            is_month_end = date.day >= 25
            
            # 1. Enhance lag features with historical patterns
            for lag in [1, 2, 7]:
                col = f'{model_type}_lag_{lag}'
                if col in input_df.columns:
                    if i >= lag:
                        # Start with previous prediction
                        base_value = predictions[i-lag]
                        
                        # Apply day-of-week adjustment from historical patterns
                        if hist_stats and 'daily' in hist_stats and day_of_week in hist_stats['daily']:
                            day_mean = hist_stats['daily'][day_of_week]['mean']
                            day_median = hist_stats['daily'][day_of_week]['median']
                            if day_mean > 0:
                                # Adjust towards historical pattern for this day
                                adjustment = 0.3 * (day_median/day_mean) + 0.7
                                base_value *= adjustment
                        
                        # Add noise based on historical variability
                        if hist_stats and 'overall_std' in hist_stats and hist_stats['overall_std'] > 0:
                            noise = np.random.normal(0, hist_stats['overall_std'] * 0.3)
                            base_value = max(0, base_value + noise)
                        
                        input_df[col] = base_value
                    elif len(predictions) >= lag:
                        input_df[col] = predictions[-lag]
                    elif len(history) > 0:
                        # Use historical value with day-of-week adjustment
                        hist_val = history.iloc[-lag][target_col] if len(history) >= lag else history[target_col].median()
                        input_df[col] = hist_val
            
            # 2. Make the base prediction
            input_df = input_df.reindex(columns=model_features[model_type], fill_value=0)
            pred = model.predict(input_df)[0]
            
            # 3. Post-processing to match historical variability
            if hist_stats and len(history) > 0:
                # Adjust prediction based on day-of-week pattern
                if 'daily' in hist_stats and day_of_week in hist_stats['daily']:
                    day_median = hist_stats['daily'][day_of_week]['median']
                    day_max = hist_stats['daily'][day_of_week]['max']
                    
                    # Blend prediction with historical pattern (70% prediction, 30% historical pattern)
                    if day_median > 0:
                        pred = 0.7 * pred + 0.3 * day_median
                    
                    # Add some of the "peakiness" from historical data
                    peak_factor = min(2.0, day_max / max(1, day_median))
                    pred *= (0.9 + 0.2 * peak_factor)
                
                # Add special day boosts
                if is_weekend:
                    pred *= 1.5  # Stronger weekend effect
                if is_holiday:
                    pred *= 1.8  # Stronger holiday effect
                if is_month_end:
                    pred *= 1.3  # Month-end effect
                
                # Add controlled noise
                if 'overall_std' in hist_stats and hist_stats['overall_std'] > 0:
                    noise = np.random.normal(0, hist_stats['overall_std'] * 0.4)
                    pred = max(0, pred + noise)
            
            predictions.append(pred)
            recent_values.append(pred)
            if len(recent_values) > 7:
                recent_values.pop(0)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot historical data if available
        if len(history) > 0:
            hist_to_show = history[history['fk_jour'] >= (start_date - timedelta(days=180))]
            ax.plot(hist_to_show['fk_jour'], hist_to_show[target_col], 
                    'b-', label='Historique', linewidth=1.5, alpha=0.7)
        
        # Plot predictions
        ax.plot(dates, predictions, 'r-', label='Predictions', linewidth=2)
        
        # Add visual indicators for special days
        for date in dates:
            if date.weekday() >= 5:  # Weekends
                ax.axvspan(date, date + timedelta(days=1), color='yellow', alpha=0.1)
            if date.date() in holidays.France():  # Holidays
                ax.axvline(date, color='green', linestyle=':', alpha=0.5, linewidth=1)
        
        # Formatting
        titles = {
            'qte': 'Quantity sold',
            'ticket': 'Tickets Count',
            'ca_ttc': 'Revenue (Dinars)'
        }
        ax.set_title(f"{titles.get(model_type, 'Predictions')} - Article {article_id}", fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(titles.get(model_type, 'Value'), fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
        plt.xticks(rotation=45)
        fig.tight_layout()
        
        # Save plot
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return f"data:image/png;base64,{plot_data}"
    
    except Exception as e:
        print(f"Error generating plot: {str(e)}")
        traceback.print_exc()
        return None 

# Load models with comprehensive error handling
try:
    print("\nLoading models...")
    models_dir = 'models'
    
    def load_model_package(filename):
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)
    
    qte_pkg = load_model_package('best_qte_model.joblib')
    ticket_pkg = load_model_package('optimized_ticket_model.joblib')
    ca_ttc_pkg = load_model_package('improved_ca_ttc_model.joblib')
    
    # Extract models
    models = {
        'qte': qte_pkg['model'],
        'ticket': ticket_pkg['model'],
        'ca_ttc': ca_ttc_pkg['model']
    }
    
    # Get feature names - handle different storage formats
    def get_features(pkg):
        if 'metadata' in pkg and 'features' in pkg['metadata']:
            return pkg['metadata']['features']
        elif 'features' in pkg:
            return pkg['features']
        elif hasattr(pkg['model'], 'feature_names_in_'):
            return list(pkg['model'].feature_names_in_)
        else:
            # Fallback: try to infer from the model's coefficients or feature importances
            try:
                if hasattr(pkg['model'], 'coef_'):
                    return [f'feature_{i}' for i in range(len(pkg['model'].coef_))]
                elif hasattr(pkg['model'], 'feature_importances_'):
                    return [f'feature_{i}' for i in range(len(pkg['model'].feature_importances_))]
                else:
                    raise ValueError("Could not determine features from model package")
            except:
                raise ValueError("Could not determine features from model package")
    
    model_features = {
        'qte': get_features(qte_pkg),
        'ticket': get_features(ticket_pkg),
        'ca_ttc': get_features(ca_ttc_pkg)
    }
    
    print("\nModels loaded successfully!")
    print("Available models:", list(models.keys()))
    print("\nModel features:")
    for model_type, features in model_features.items():
        print(f"{model_type}: {features}")
    
except Exception as e:
    print(f"\nERROR loading models: {str(e)}")
    traceback.print_exc()
    models = {}
    model_features = {}

@app.route('/', methods=['GET', 'POST'])
def home():
    # Calculate basic counts from historical data
    active_products_count = historical_df['fk_article'].nunique()
    active_stores_count = historical_df['fk_magasin'].nunique()
    
    # Get the most recent date in your historical data (2018)
    most_recent_date = historical_df['fk_jour'].max()
    
    # Calculate statistics based on the last 30 days of 2018 data
    last_30_days = historical_df[historical_df['fk_jour'] >= (most_recent_date - timedelta(days=30))]
    
    # Corrected average daily sales calculation - sum per day first
    daily_sales = last_30_days.groupby('fk_jour')['ca_ttc'].sum()
    avg_daily_sales = float(daily_sales.mean()) if not daily_sales.empty else 0.0
    
    # Weekly trend calculation using the last 2 weeks of 2018
    current_week = historical_df[historical_df['fk_jour'] >= (most_recent_date - timedelta(days=7))]
    previous_week = historical_df[
        (historical_df['fk_jour'] >= (most_recent_date - timedelta(days=14))) &
        (historical_df['fk_jour'] < (most_recent_date - timedelta(days=7)))
    ]
    
    # Calculate weekly sums correctly
    current_week_sales = current_week.groupby('fk_jour')['ca_ttc'].sum().sum()
    previous_week_sales = previous_week.groupby('fk_jour')['ca_ttc'].sum().sum()
    sales_trend = ((current_week_sales - previous_week_sales) / previous_week_sales * 100) if previous_week_sales > 0 else 0
    
    # Best performing product (all time)
    top_product_info = historical_df.groupby('fk_article').agg({
        'ca_ttc': 'sum',
        'qte': 'sum'
    }).sort_values('ca_ttc', ascending=False).iloc[0]

    # Common template variables for both GET and POST
    template_vars = {
        'active_products_count': active_products_count,
        'active_stores_count': active_stores_count,
        'avg_daily_sales': avg_daily_sales,  # Keep as float for template formatting
        'sales_trend': sales_trend,
        'top_product_id': top_product_info.name,
        'top_product_sales': float(top_product_info['ca_ttc']),
        'top_product_quantity': int(top_product_info['qte']),
        'data_year': most_recent_date.year  # Show the year of the historical data
    }

    if request.method == 'POST':
        try:
            model_type = request.form['model_type']
            if model_type not in models:
                raise ValueError(f"Model {model_type} not available")
            
            model = models[model_type]
            input_df = create_all_features(request.form)
            
            # Get exact features the model expects
            required_features = model_features[model_type]
            
            # Verify feature alignment
            missing = set(required_features) - set(input_df.columns)
            extra = set(input_df.columns) - set(required_features)
            
            if missing:
                raise ValueError(f"Missing features: {missing}")
            
            # Select only required features in correct order
            input_df = input_df[required_features]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Generate time series plot
            article_id = int(request.form['article_id'])
            magasin_id = int(request.form['magasin_id'])
            plot_url = generate_time_series_plot(
                article_id=int(request.form['article_id']),
                model_type=model_type,
                model=model,
                start_date=request.form['date']
            )
            
            # Format the prediction based on model type
            if model_type in ['qte', 'ticket']:
                # Round to nearest integer for quantity and ticket count
                prediction = max(0, round(prediction))  # Ensure non-negative
                formatted_pred = f"{prediction}"
            else:
                # Keep 2 decimals for revenue, ensure non-negative
                prediction = max(0, prediction)
                formatted_pred = f"{prediction:.2f} Dinars"
            
            # Add POST-specific variables to template vars
            template_vars.update({
                'prediction': formatted_pred,
                'plot_url': plot_url if plot_url else None,
                'model_type': model_type.upper(),
                'article_id': request.form['article_id'],
                'magasin_id': request.form['magasin_id'],
                'date': request.form['date']
            })
            
            return render_template('index.html', **template_vars)
        
        except Exception as e:
            # Add error variables to template vars
            template_vars.update({
                'error': f"Prediction error: {str(e)}",
                'article_id': request.form.get('article_id', ''),
                'magasin_id': request.form.get('magasin_id', ''),
                'date': request.form.get('date', '')
            })
            return render_template('index.html', **template_vars)
    
    # For GET requests
    return render_template('index.html', **template_vars)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)