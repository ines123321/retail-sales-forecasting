# Sales Forecasting Web Application

##  Project Overview
A machine learning-powered sales forecasting web application built with Flask that predicts product sales quantities, ticket counts, and revenue for retail stores. The application uses historical sales data and temporal features to generate accurate forecasts with visual time-series predictions.

##  Features

### Core Functionality
- **Multi-model Forecasting**: Predicts three key metrics:
  - Product Quantity (`qte`) - Units sold
  - Ticket Count (`ticket`) - Number of transactions
  - Revenue (`ca_ttc`) - Turnover in Dinars
- **Interactive Time-Series Visualization**: Generates 210-day forecasts with historical comparison
- **Smart Feature Engineering**: Automatically creates lag features, temporal patterns, and holiday effects
- **Historical Data Integration**: Uses past sales patterns to enhance prediction accuracy

### Web Interface
- Clean, responsive dashboard
- Real-time prediction results
- Interactive plots with zoomable timelines
- Key performance indicators display
- Error handling with user-friendly messages

##  Project Structure
```
├── app.py                 
├── models/                   
│   ├── best_qte_model.joblib
│   ├── optimized_ticket_model.joblib
│   └── improved_ca_ttc_model.joblib
├── dataset_final_nettoye.csv 
├── templates/             
├── static/images/         
├── requirements.txt        
├── Dockerfile            
└── .dockerignore        
```

##  Installation

### Prerequisites
- Python 3.8+
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd sales-forecasting-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```
   The app will be available at `http://localhost:5000`

### Docker Deployment
```bash
# Build the Docker image
docker build -t sales-forecasting-app .

# Run the container
docker run -p 5000:5000 sales-forecasting-app
```

##  Usage Guide

### Making Predictions
1. Open the web application in your browser
2. Enter the required parameters:
   - **Article ID**: Product identifier
   - **Store ID**: Store identifier
   - **Prediction Date**: Date for the forecast
   - **Model Type**: Select prediction type (Quantity/Tickets/Revenue)
3. Click "Generate Prediction"
4. View results including:
   - Numerical prediction value
   - 210-day forecast chart
   - Historical comparison

### Understanding the Output
- **Quantity predictions** are rounded to nearest integer
- **Revenue predictions** are shown in Dinars with 2 decimal places
- **Charts show**:
  - Blue line: Historical sales (last 180 days)
  - Red line: Future predictions (210 days)
  - Yellow highlights: Weekends
  - Green dotted lines: French holidays

##  Technical Details

### Feature Engineering
The application creates comprehensive features including:
- **Temporal Features**: Day of week, month, holidays, weekends
- **Lag Features**: Sales from previous 1, 2, and 7 days
- **Rolling Statistics**: 7-day averages and maximums
- **Cyclical Patterns**: Sine/cosine transformations for seasonal trends
- **Store/Product Specific**: Individual store and product historical averages

### Machine Learning Models
- Pre-trained scikit-learn models stored as `.joblib` files
- Each model optimized for specific prediction type
- Automatic feature alignment and validation
- Fallback mechanisms for missing historical data

##  Development

### Adding New Models
1. Train your model using the same feature set
2. Save with `joblib.dump()`
3. Update `app.py` to load the new model
4. Add to the `models` dictionary

### Extending Features
Modify the `create_all_features()` function to add:
- New temporal patterns
- Additional lag periods
- Custom business rules
- External data sources

##  Data Format
The application expects historical data in CSV format with columns:
- `fk_jour`: Date (format: DD/MM/YYYY)
- `fk_article`: Article ID
- `fk_magasin`: Store ID
- `qte`: Quantity sold
- `nbr_ticket`: Number of tickets
- `ca_ttc`: Revenue

## Tech Stack
- **Backend**: Flask
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib
- **Data Processing**: pandas, numpy
- **Deployment**: Docker
- 
## Limitations & Future Improvements
- Models are trained on historical data from 2018 only
- No automatic retraining pipeline yet
- External factors (promotions, weather) not included
- Future work: add model retraining, API endpoints, and confidence intervals


---

**Note**: This application uses historical data from 2018 for training and demonstration purposes. For production use, update with recent data and retrain models accordingly.
