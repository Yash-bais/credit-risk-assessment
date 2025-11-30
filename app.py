from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for models and scaler
models = {}
scaler = None
feature_columns = None

def load_or_train_models():
    """Load saved models or train new ones if they don't exist"""
    global models, scaler, feature_columns
    
    if (os.path.exists('models/rfc_model.pkl') and 
        os.path.exists('models/gbc_model.pkl') and 
        os.path.exists('models/scaler.pkl')):
        # Load existing models
        print("Loading existing models...")
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/rfc_model.pkl', 'rb') as f:
            models['random_forest'] = pickle.load(f)
        with open('models/gbc_model.pkl', 'rb') as f:
            models['gradient_boosting'] = pickle.load(f)
        with open('models/svc_model.pkl', 'rb') as f:
            models['svc'] = pickle.load(f)
        with open('models/lr_model.pkl', 'rb') as f:
            models['logistic_regression'] = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        print(f"✅ Models loaded successfully! Using {len(feature_columns)} features.")
    else:
        # Train new models
        print("Models not found. Training new models...")
        train_models()

def feature_engineering(df):
    """Create additional features to improve model performance"""
    df = df.copy()
    
    # Create new features
    # Total debt
    df['total_debt'] = df['creddebt'] + df['othdebt']
    
    # Debt to income ratio (already exists but ensure it's calculated correctly)
    df['debt_income_ratio'] = df['debtinc']
    
    # Income per year of employment (stability indicator)
    df['income_per_employment_year'] = df['income'] / (df['employ'] + 1)  # +1 to avoid division by zero
    
    # Age groups (categorical encoding)
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 100], labels=[1, 2, 3, 4])
    df['age_group'] = df['age_group'].astype(float)
    
    # High debt indicator
    df['high_debt'] = (df['debtinc'] > df['debtinc'].median()).astype(float)
    
    # Credit utilization (if we had credit limit, but using debt/income as proxy)
    df['credit_utilization'] = df['creddebt'] / (df['income'] / 12 + 1) * 100
    
    # Employment stability (years employed relative to age)
    df['employment_stability'] = df['employ'] / (df['age'] - 18 + 1)  # Assuming work starts at 18
    
    # Address stability (years at address)
    df['address_stability'] = df['address']
    
    # Income level category
    df['income_level'] = pd.cut(df['income'], bins=[0, 30, 60, 100, 200, 1000], labels=[1, 2, 3, 4, 5])
    df['income_level'] = df['income_level'].astype(float)
    
    return df

def train_models():
    """Train improved models on the existing dataset"""
    global models, scaler, feature_columns
    
    print("Loading and preprocessing dataset...")
    # Load dataset
    df = pd.read_csv('bankloans.csv')
    
    # Drop rows with missing target values
    df = df.dropna(subset=['default'])
    
    print(f"Dataset shape after dropping missing targets: {df.shape}")
    print(f"Class distribution:\n{df['default'].value_counts()}")
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Features and target
    # Keep original features plus new engineered features
    feature_cols = ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt',
                    'total_debt', 'income_per_employment_year', 'age_group', 'high_debt',
                    'credit_utilization', 'employment_stability', 'address_stability', 'income_level']
    
    # Only use columns that exist
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols]
    y = df['default'].astype(int)
    feature_columns = X.columns.tolist()
    
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance with SMOTE
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"After SMOTE - Training set: {X_train_balanced.shape}")
    print(f"Balanced class distribution:\n{pd.Series(y_train_balanced).value_counts()}")
    
    # Train improved models with better hyperparameters
    
    # 1. Random Forest with tuned parameters
    print("\nTraining Random Forest Classifier...")
    rfc = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rfc.fit(X_train_balanced, y_train_balanced)
    rfc_score = rfc.score(X_test_scaled, y_test)
    rfc_pred = rfc.predict(X_test_scaled)
    rfc_auc = roc_auc_score(y_test, rfc.predict_proba(X_test_scaled)[:, 1])
    print(f"Random Forest - Accuracy: {rfc_score:.4f}, AUC: {rfc_auc:.4f}")
    
    # 2. Gradient Boosting Classifier (better than basic RF)
    print("\nTraining Gradient Boosting Classifier...")
    gbc = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42
    )
    gbc.fit(X_train_balanced, y_train_balanced)
    gbc_score = gbc.score(X_test_scaled, y_test)
    gbc_pred = gbc.predict(X_test_scaled)
    gbc_auc = roc_auc_score(y_test, gbc.predict_proba(X_test_scaled)[:, 1])
    print(f"Gradient Boosting - Accuracy: {gbc_score:.4f}, AUC: {gbc_auc:.4f}")
    
    # 3. SVM with better parameters
    print("\nTraining Support Vector Classifier...")
    svc = SVC(
        C=1.0,
        gamma='scale',
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    svc.fit(X_train_balanced, y_train_balanced)
    svc_score = svc.score(X_test_scaled, y_test)
    svc_pred = svc.predict(X_test_scaled)
    svc_auc = roc_auc_score(y_test, svc.predict_proba(X_test_scaled)[:, 1])
    print(f"SVM - Accuracy: {svc_score:.4f}, AUC: {svc_auc:.4f}")
    
    # 4. Logistic Regression with regularization
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        C=0.1,
        penalty='l2',
        solver='lbfgs',
        max_iter=2000,
        class_weight='balanced',
        random_state=42
    )
    lr.fit(X_train_balanced, y_train_balanced)
    lr_score = lr.score(X_test_scaled, y_test)
    lr_pred = lr.predict(X_test_scaled)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
    print(f"Logistic Regression - Accuracy: {lr_score:.4f}, AUC: {lr_auc:.4f}")
    
    # Print detailed evaluation
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':<12} {'AUC Score':<12}")
    print("-"*60)
    print(f"{'Random Forest':<25} {rfc_score:<12.4f} {rfc_auc:<12.4f}")
    print(f"{'Gradient Boosting':<25} {gbc_score:<12.4f} {gbc_auc:<12.4f}")
    print(f"{'SVM':<25} {svc_score:<12.4f} {svc_auc:<12.4f}")
    print(f"{'Logistic Regression':<25} {lr_score:<12.4f} {lr_auc:<12.4f}")
    print("="*60)
    
    models = {
        'random_forest': rfc,
        'gradient_boosting': gbc,
        'svc': svc,
        'logistic_regression': lr
    }
    
    # Save models
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/rfc_model.pkl', 'wb') as f:
        pickle.dump(rfc, f)
    with open('models/gbc_model.pkl', 'wb') as f:
        pickle.dump(gbc, f)
    with open('models/svc_model.pkl', 'wb') as f:
        pickle.dump(svc, f)
    with open('models/lr_model.pkl', 'wb') as f:
        pickle.dump(lr, f)
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print("\n✅ Models trained and saved successfully!")
    print(f"Best model: Gradient Boosting (Accuracy: {gbc_score:.4f}, AUC: {gbc_auc:.4f})")

def map_comprehensive_features_to_model_features(form_data):
    """Map form features to model's expected features with engineering"""
    # Extract values from form
    age = float(form_data.get('age', 0))
    annual_income = float(form_data.get('annual_income', 0))
    years_employed = float(form_data.get('years_employed', 0))
    credit_history_years = float(form_data.get('credit_history_years', 0))
    existing_debt = float(form_data.get('existing_debt', 0))
    home_ownership = form_data.get('home_ownership', 'rent')
    loan_amount = float(form_data.get('loan_amount', 0))
    credit_score = float(form_data.get('credit_score', 650))
    
    # Map to model features (based on bankloans.csv structure)
    # The model expects: age, ed, employ, address, income, debtinc, creddebt, othdebt
    
    # Education (ed): Estimate based on employment status and income
    # Since we don't have education directly, use a default value of 2 (some college)
    ed = 2
    
    # Employment years (employ)
    employ = years_employed
    
    # Address stability (address): Estimate based on home ownership and credit history
    # Own/Mortgage = more stable, Rent = less stable
    if home_ownership in ['own', 'mortgage']:
        address = max(credit_history_years, 5)  # At least 5 years if owns home
    else:
        address = min(credit_history_years, 3)  # Max 3 years if renting
    
    # Income
    income = annual_income
    
    # Debt-to-Income Ratio (debtinc)
    # Calculate monthly debt payment (assume existing_debt is annual debt payment)
    monthly_debt = existing_debt / 12 if existing_debt > 0 else 0
    monthly_income = annual_income / 12 if annual_income > 0 else 1
    debtinc = (monthly_debt / monthly_income * 100) if monthly_income > 0 else 0
    
    # Credit debt (creddebt): Estimate as 60% of existing debt
    creddebt = existing_debt * 0.6 if existing_debt > 0 else 0
    
    # Other debt (othdebt): Remaining 40% of existing debt
    othdebt = existing_debt * 0.4 if existing_debt > 0 else 0
    
    # Create base features array
    base_features = {
        'age': age,
        'ed': ed,
        'employ': employ,
        'address': address,
        'income': income,
        'debtinc': debtinc,
        'creddebt': creddebt,
        'othdebt': othdebt
    }
    
    # Add engineered features (same as in training)
    base_features['total_debt'] = creddebt + othdebt
    base_features['income_per_employment_year'] = income / (employ + 1)
    
    # Age groups
    if age <= 30:
        age_group = 1
    elif age <= 40:
        age_group = 2
    elif age <= 50:
        age_group = 3
    else:
        age_group = 4
    base_features['age_group'] = float(age_group)
    
    # High debt indicator (using median threshold from training - approximate)
    base_features['high_debt'] = 1.0 if debtinc > 10.0 else 0.0
    
    # Credit utilization
    base_features['credit_utilization'] = (creddebt / (income / 12 + 1)) * 100
    
    # Employment stability
    base_features['employment_stability'] = employ / (age - 18 + 1) if age > 18 else 0
    
    # Address stability (same as address)
    base_features['address_stability'] = address
    
    # Income level
    if income <= 30:
        income_level = 1
    elif income <= 60:
        income_level = 2
    elif income <= 100:
        income_level = 3
    elif income <= 200:
        income_level = 4
    else:
        income_level = 5
    base_features['income_level'] = float(income_level)
    
    # Create feature array in the order expected by the model
    feature_order = ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt',
                    'total_debt', 'income_per_employment_year', 'age_group', 'high_debt',
                    'credit_utilization', 'employment_stability', 'address_stability', 'income_level']
    
    # Only include features that exist in the model
    features_list = [base_features.get(col, 0) for col in feature_columns if col in base_features]
    
    # Ensure we have the right number of features
    if len(features_list) != len(feature_columns):
        # Fallback: use base features only
        base_order = ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt']
        features_list = [base_features.get(col, 0) for col in base_order]
    
    features = np.array([features_list])
    
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form
        
        # Map comprehensive features to model features
        features = map_comprehensive_features_to_model_features(form_data)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for model_name, model in models.items():
            try:
                pred = model.predict(features_scaled)[0]
                prob = model.predict_proba(features_scaled)[0]
                
                predictions[model_name] = int(pred)
                probabilities[model_name] = {
                    'no_default': float(prob[0]) * 100,
                    'default': float(prob[1]) * 100
                }
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                # Fallback prediction
                predictions[model_name] = 0
                probabilities[model_name] = {'no_default': 50.0, 'default': 50.0}
        
        # Calculate ensemble prediction (majority vote)
        pred_values = list(predictions.values())
        ensemble_pred = 1 if sum(pred_values) >= 2 else 0
        
        # Calculate average probability
        avg_no_default = np.mean([prob['no_default'] for prob in probabilities.values()])
        avg_default = np.mean([prob['default'] for prob in probabilities.values()])
        
        # Calculate credit score from form data
        credit_score = float(form_data.get('credit_score', 650))
        
        # Calculate predicted score (0-100 scale based on probability of no default)
        predicted_score = round(avg_no_default, 1)
        
        # Predict loan grade based on credit score and default probability
        # Grade calculation: A (best) to G (worst)
        def predict_grade(credit_score, default_prob):
            # Combine credit score (normalized to 0-100) and default probability
            # Higher credit score and lower default prob = better grade
            credit_factor = (credit_score - 300) / 550  # Normalize 300-850 to 0-1
            risk_factor = 1 - (default_prob / 100)  # Lower default prob = higher factor
            
            # Combined score (0-100)
            combined_score = (credit_factor * 0.6 + risk_factor * 0.4) * 100
            
            # Grade mapping
            if combined_score >= 85:
                return 'A'
            elif combined_score >= 75:
                return 'B'
            elif combined_score >= 65:
                return 'C'
            elif combined_score >= 55:
                return 'D'
            elif combined_score >= 45:
                return 'E'
            elif combined_score >= 35:
                return 'F'
            else:
                return 'G'
        
        predicted_grade = predict_grade(credit_score, avg_default)
        
        # Format probabilities for display
        formatted_probabilities = {}
        for model_name, prob_dict in probabilities.items():
            formatted_probabilities[model_name] = {
                'no_default': round(prob_dict['no_default'], 2),
                'default': round(prob_dict['default'], 2)
            }
        
        # Format form data for display
        formatted_form_data = {}
        for key, value in form_data.items():
            if key in ['annual_income', 'loan_amount', 'existing_debt']:
                try:
                    formatted_form_data[key] = f"{float(value):,.0f}"
                except:
                    formatted_form_data[key] = value
            else:
                formatted_form_data[key] = value
        
        # Prepare results
        results = {
            'ensemble_prediction': ensemble_pred,
            'ensemble_probability': {
                'no_default': round(avg_no_default, 2),
                'default': round(avg_default, 2)
            },
            'individual_predictions': predictions,
            'individual_probabilities': formatted_probabilities,
            'risk_level': 'High Risk' if ensemble_pred == 1 else 'Low Risk',
            'recommendation': 'Loan Application Rejected' if ensemble_pred == 1 else 'Loan Application Approved',
            'predicted_score': predicted_score,
            'predicted_grade': predicted_grade
        }
        
        return render_template('result.html', results=results, form_data=formatted_form_data)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON predictions"""
    try:
        data = request.json
        features = map_comprehensive_features_to_model_features(data)
        features_scaled = scaler.transform(features)
        
        predictions = {}
        probabilities = {}
        
        for model_name, model in models.items():
            pred = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0]
            
            predictions[model_name] = int(pred)
            probabilities[model_name] = {
                'no_default': float(prob[0]) * 100,
                'default': float(prob[1]) * 100
            }
        
        pred_values = list(predictions.values())
        ensemble_pred = 1 if sum(pred_values) >= 2 else 0
        
        # Calculate average probability
        avg_no_default = np.mean([prob['no_default'] for prob in probabilities.values()])
        avg_default = np.mean([prob['default'] for prob in probabilities.values()])
        
        # Calculate predicted score and grade
        credit_score = float(data.get('credit_score', 650))
        predicted_score = round(avg_no_default, 1)
        
        # Grade prediction function (same as in predict route)
        def predict_grade(credit_score, default_prob):
            credit_factor = (credit_score - 300) / 550
            risk_factor = 1 - (default_prob / 100)
            combined_score = (credit_factor * 0.6 + risk_factor * 0.4) * 100
            
            if combined_score >= 85:
                return 'A'
            elif combined_score >= 75:
                return 'B'
            elif combined_score >= 65:
                return 'C'
            elif combined_score >= 55:
                return 'D'
            elif combined_score >= 45:
                return 'E'
            elif combined_score >= 35:
                return 'F'
            else:
                return 'G'
        
        predicted_grade = predict_grade(credit_score, avg_default)
        
        return jsonify({
            'prediction': ensemble_pred,
            'risk_level': 'High Risk' if ensemble_pred == 1 else 'Low Risk',
            'probabilities': probabilities,
            'predictions': predictions,
            'predicted_score': predicted_score,
            'predicted_grade': predicted_grade,
            'ensemble_probability': {
                'no_default': round(avg_no_default, 2),
                'default': round(avg_default, 2)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Loading models...")
    load_or_train_models()
    print("Models loaded. Starting Flask app...")
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # For production (Gunicorn)
    print("Loading models for production...")
    load_or_train_models()
    print("Models loaded. Ready for production.")

