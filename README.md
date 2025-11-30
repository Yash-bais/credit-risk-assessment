# Credit Risk Assessment System

A Flask web application for predicting loan default risk using machine learning models.

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

## Installation

1. **Activate your virtual environment** (if using one):
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Method 1: Direct Python Execution

```bash
python app.py
```

Or on Windows with virtual environment:
```bash
.venv\Scripts\python.exe app.py
```

### Method 2: Using Flask Command

```bash
flask run
```

Or with specific host and port:
```bash
flask run --host=0.0.0.0 --port=5000
```

## Accessing the Application

Once the server starts, you'll see output like:
```
Loading models...
Models not found. Training new improved models...
...
 * Running on http://127.0.0.1:5000
```

Open your web browser and navigate to:
- **Local access**: http://localhost:5000
- **Network access**: http://127.0.0.1:5000

## First Run

On the first run, the application will:
1. Load the dataset (`bankloans.csv`)
2. Perform feature engineering
3. Train 4 machine learning models:
   - Random Forest
   - Gradient Boosting
   - Support Vector Machine (SVM)
   - Logistic Regression
4. Save the trained models to the `models/` directory

This may take a few minutes. Subsequent runs will load the saved models much faster.

## Using the Application

1. Fill out the loan application form with:
   - Personal Information (Name, Email, Age)
   - Employment & Income details
   - Credit Information
   - Loan Details
   - Financial Obligations

2. Click "Check Eligibility →"

3. View the results showing:
   - Loan approval/rejection recommendation
   - Risk level (High/Low)
   - Predicted Score (0-100)
   - Predicted Grade (A-G)
   - Individual model predictions
   - Probability breakdowns

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, change it in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
```

### Missing Dependencies
If you get import errors, install missing packages:
```bash
pip install -r requirements.txt
```

### Model Training Errors
If models fail to train, check:
- `bankloans.csv` exists in the project directory
- Dataset has the required columns
- Sufficient memory available

## Project Structure

```
technical/
├── app.py                 # Main Flask application
├── bankloans.csv         # Dataset
├── requirements.txt       # Python dependencies
├── models/               # Saved trained models (created after first run)
├── templates/            # HTML templates
│   ├── index.html       # Main form page
│   ├── result.html      # Results page
│   └── error.html       # Error page
└── static/               # Static files
    └── style.css        # CSS styling
```

## Features

- **4 ML Models**: Ensemble of Random Forest, Gradient Boosting, SVM, and Logistic Regression
- **Feature Engineering**: 8 additional engineered features for better accuracy
- **Class Imbalance Handling**: SMOTE oversampling for balanced predictions
- **Real-time Predictions**: Instant risk assessment
- **Score & Grade Prediction**: Automatic loan grade assignment (A-G)
- **Modern UI**: Rich color scheme and responsive design

## Model Performance

The improved models include:
- Feature engineering (8 new features)
- SMOTE for handling class imbalance
- Optimized hyperparameters
- Better evaluation metrics (AUC-ROC)

## License

This project is for educational purposes.

