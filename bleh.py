import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, PoissonRegressor, LogisticRegression
)
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# --- Constants ---
# Using the years defined in your original features.py and models.py
validation_year = 2012
test_year = 2016 

# --- Loss Functions ---
sq_loss      = lambda true_data, predict_data : 1/len(true_data) * np.sum((true_data - predict_data)**2)
avg_std_loss = lambda true_data, predict_data : np.sqrt(sq_loss(true_data, predict_data))

# --- Regression Class (from regression.py) ---
class Regressor:
    """Class definition for various Regression models."""
    def __init__(self, model_type='LinearRegression'):
        self.model_type = model_type
        
        if model_type == 'LinearReg':
            self.model = LinearRegression(fit_intercept=True)
        elif model_type == 'Ridge':
            self.model = Ridge(fit_intercept=True)
        elif model_type == 'Lasso':
            self.model = Lasso(fit_intercept=True)
        elif model_type == 'SVR':
            self.model = SVR(kernel='poly', degree=3, gamma='scale', epsilon=0.1, C=1.0, max_iter=1000)
        elif model_type == 'Poisson':
            self.model = PoissonRegressor(max_iter=10000)
        elif model_type == 'RandomForest':
            self.model = RandomForestRegressor(max_depth=5, n_estimators=20)

    def fit(self, x, y):
        self.model.fit(x, y)
    
    def predict(self, x):
        return self.model.predict(x)

    def fit_cv(self, x, y):
        # Hyperparameter tuning logic (similar to regression.py)
        if self.model_type == 'Ridge':
            params = {'alpha':np.logspace(-6, 6, 13)}
            self.model_cv = GridSearchCV(estimator=self.model, param_grid=params, scoring='neg_mean_squared_error').fit(x, y)
        elif self.model_type == 'Lasso':
            params = {'alpha':np.logspace(-6, 6, 13)}
            self.model_cv = GridSearchCV(estimator=self.model, param_grid=params, scoring='neg_mean_squared_error').fit(x, y)
        elif self.model_type == 'SVR':
            params = {'kernel': ['rbf', 'poly'], 'gamma': ['scale', 'auto'], 'C': [0.1, 1, 10]}
            self.model_cv = GridSearchCV(estimator=self.model, param_grid=params, scoring='neg_mean_squared_error').fit(x, y)
        elif self.model_type == 'Poisson':
            params = {'alpha':np.linspace(0.01,10,100), 'max_iter':[100000]}
            self.model_cv = GridSearchCV(estimator=self.model, param_grid=params, scoring='neg_mean_squared_error').fit(x,y)
        elif self.model_type == 'RandomForest':
            params = {'bootstrap': [True, False], 'max_depth': [10, 30, 50, None], 'n_estimators': [200, 400]}
            self.model_cv = RandomizedSearchCV(estimator=self.model, param_distributions=params, n_iter=20, cv=3, verbose=1, random_state=42).fit(x, y)
        # FIX: LinearReg has no CV parameters, so we fit the model and assign it to model_cv
        elif self.model_type == 'LinearReg': 
            self.model.fit(x, y)
            self.model_cv = self.model


# --- Classifier Class (from classifier.py) ---
class Classifier:
    """Class definition for various Classification models."""
    def __init__(self, model_type='Logistic_Reg'):
        self.model_type = model_type
        if model_type == 'Logistic_Reg':
            self.model = LogisticRegression(max_iter=10000)
        elif model_type == 'SVC':
            self.model = SVC(kernel="rbf", C=0.025)
        elif model_type == 'RandomForest':
            self.model = RandomForestClassifier(max_depth=5, n_estimators=20)
        elif model_type == 'GaussianNB':
            self.model = GaussianNB()
        elif model_type == 'MLP':
            self.model = MLPClassifier(alpha=1, max_iter=1000)

    def fit(self, x, y):
        self.model.fit(x, y)
    
    def predict(self, x):
        return self.model.predict(x)

    def fit_cv(self, x, y):
        # Hyperparameter tuning logic (similar to classifier.py)
        if self.model_type == 'Logistic_Reg':
            params = {'C': np.logspace(-4, 4, 20)}
            self.model_cv = GridSearchCV(estimator=self.model, param_grid=params, verbose=1, scoring='accuracy').fit(x, y)
        elif self.model_type == 'SVC':
            params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
            self.model_cv = GridSearchCV(estimator=self.model, param_grid=params, verbose=1, scoring='accuracy').fit(x, y)
        elif self.model_type == 'RandomForest':
            params = {'bootstrap': [True, False], 'max_depth': [10, 30, 50, None], 'n_estimators': [200, 400]}
            self.model_cv = RandomizedSearchCV(estimator=self.model, param_distributions=params, n_iter=20, cv=3, verbose=1, random_state=42).fit(x, y)
        elif self.model_type == 'GaussianNB':
            params = {'var_smoothing': np.logspace(0,-9, num=100)}
            self.model_cv = GridSearchCV(estimator=self.model, param_grid=params, verbose=1, scoring='accuracy').fit(x, y)
        # FIX: MLP has no CV parameters defined, so we fit the model and assign it to model_cv
        elif self.model_type == 'MLP':
            self.model.fit(x, y)
            self.model_cv = self.model

# --- Plotting Functions (from graphs.py, simplified for display) ---
def plot(x, y, ttl=None, line=False):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel('True Medals')
    plt.ylabel('Predicted Medals')
    if line:
        lims = [min(min(x), min(y)), max(max(x), max(y))]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    plt.title(ttl if ttl else 'Scatter Plot')
    plt.show()

def plot_clf(clf_perf):
    plt.figure()
    plt.bar(range(len(clf_perf)), list(clf_perf.values()), align='center')
    plt.xticks(range(len(clf_perf)), list(clf_perf.keys()), rotation=45, ha="right")
    plt.xlabel('Classifier Algorithm')
    plt.ylabel('Accuracy')
    plt.title('Classifier Performance')
    plt.tight_layout()
    plt.show()

def plot_reg(reg_perf):
    X_axis = np.arange(len(reg_perf))
    total_std_dev = [array[0] for array in reg_perf.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(X_axis, total_std_dev, 0.4, label = 'All Countries')
    plt.xticks(range(len(reg_perf)), list(reg_perf.keys()), rotation=45, ha="right")
    plt.ylim([0, 25])
    plt.xlabel('Regressor')
    plt.ylabel('Average Std Dev [Medals]')
    plt.title('Regression Algorithm Performance')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Data Cleaning and Feature Engineering (Modified from cleaned.py) ---
def clean_and_feature_engineer(data):
    """
    Cleans the raw Olympic data and engineers features using only columns from the CSV.
    """
    df = data.copy()
    df = df[df.Season == 'Summer'].copy()
    
    # Handle team medals duplication: Deduplicate to count team medals once
    df_dedup = df.drop_duplicates(subset=['Event', 'Year', 'Team', 'Medal']).copy()
    
    # Target: Binary/Count for Medals
    df_dedup['Medal_Won'] = df_dedup['Medal'].apply(lambda x: 0 if pd.isna(x) else 1)
    
    # Feature 1: Total Athletes (count unique names per NOC/Year)
    athlete_count_df = df.groupby(['NOC', 'Year'])['Name'].nunique().reset_index(name='Athletes')
    
    # Target: Total Medals (sum of Medal_Won per NOC/Year)
    medal_count_df = df_dedup.groupby(['NOC', 'Year'])['Medal_Won'].sum().reset_index(name='Medals')
    
    # Merge features and target
    agg_df = pd.merge(athlete_count_df, medal_count_df, on=['NOC', 'Year'], how='outer').fillna(0)
    
    # Feature 2: Medals from Last Games (Lag feature)
    agg_df = agg_df.sort_values(by=['NOC', 'Year'])
    agg_df['Medals_Last_Games'] = agg_df.groupby('NOC')['Medals'].shift(1).fillna(0)
    
    # Feature 3: Total Medals for that year (Global feature)
    total_medals_year = agg_df.groupby('Year')['Medals'].sum().reset_index(name='Total_Medals_Year')
    agg_df = pd.merge(agg_df, total_medals_year, on='Year', how='left')
    
    agg_df = agg_df.rename(columns={'NOC': 'Nation'})
    
    return agg_df[['Nation', 'Year', 'Athletes', 'Medals_Last_Games', 'Total_Medals_Year', 'Medals']]

# --- Data Splitting Functions (from cleaned.py) ---
def train_test_split(data, validation_year):
    training_data = data[data.Year < validation_year].copy()
    valid_data = data[data.Year == validation_year].copy()
    test_data = data[data.Year > validation_year].copy()
    
    # Features to drop for model training
    drop_cols = ['Nation', 'Medals', 'Year'] 
    
    # Create training, validation, and test sets
    x_train = training_data.drop(columns=drop_cols)
    y_train = training_data['Medals']
    
    x_valid = valid_data.drop(columns=drop_cols)
    y_valid = valid_data['Medals']

    x_test = test_data.drop(columns=drop_cols)
    y_test = test_data['Medals']
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def to_clf_data(y_train, y_valid, y_test):
    # Convert medal counts to binary classification (medal or no medal)
    yt_bool = np.zeros(len(y_train))
    yt_bool[y_train!=0] = 1
    yv_bool = np.zeros(len(y_valid))
    yv_bool[y_valid!=0] = 1
    yte_bool = np.zeros(len(y_test))
    yte_bool[y_test!=0] = 1
    
    return yt_bool, yv_bool, yte_bool

# --- Model Training and Evaluation Functions (from models.py) ---
regressor_list = ['LinearReg','Ridge','Lasso','SVR','Poisson','RandomForest']
classifier_list = ['Logistic_Reg','SVC','GaussianNB','RandomForest', 'MLP']

def train_regressors(x_t, y_t, x_v, y_v):
    reg_performance = {}
    reg_tuned = {}
    
    for reg_type in regressor_list:
        reg = Regressor(model_type=reg_type)
        print(f"Training Regressor: {reg_type}...")
        
        reg.fit_cv(x_t, y_t)
        # Use the tuned model (or base model for LinearReg) for prediction
        y_predict = reg.model_cv.predict(x_v) 
        reg_tuned[reg_type] = reg.model_cv

        y_predict[y_predict<0] = 0 # Cannot have negative medals
        
        performance = avg_std_loss(y_v, y_predict)
        reg_performance[reg_type] = (performance, 0)
        print(f"  RMSE: {performance:.2f}")

    return reg_performance, reg_tuned

def train_classifiers(x_t, y_t, x_v, y_v):
    clf_performance = {}
    clf_tuned = {}
    
    for clf_type in classifier_list:
        clf = Classifier(model_type=clf_type)
        print(f"Training Classifier: {clf_type}...")
        
        clf.fit_cv(x_t, y_t)
        y_predict = np.rint(clf.model_cv.predict(x_v))
        clf_tuned[clf_type] = clf.model_cv
            
        performance = np.sum(y_v == y_predict) / len(y_v)
        clf_performance[clf_type] = performance
        print(f"  Accuracy: {performance:.2f}")

    return clf_performance, clf_tuned

def predict_test_set(x_train_reg, y_train_reg, x_test_reg, y_test_reg, best_clf, best_reg):
    print("\n--- Testing Best Model Performance (on 2016 data) ---")
    print("Using Classification-then-Regression approach.")
    
    # 1. Classification: Predict medal/no-medal on test set
    best_clf.fit(x_train_reg, to_clf_data(y_train_reg, pd.Series([]), pd.Series([]))[0])
    y_predict_bool = best_clf.predict(x_test_reg)
    
    # 2. Filter training data for nations that won medals (to train the regressor)
    x_train_medal_winners = x_train_reg[y_train_reg != 0]
    y_train_medal_winners = y_train_reg[y_train_reg != 0]

    # 3. Filter test data for nations predicted to win medals
    x_test_medal_winners = x_test_reg[y_predict_bool == 1]
    
    # Combine feature names to ensure all model inputs are aligned
    x_train_cols = x_train_reg.columns
    x_test_medal_winners = x_test_medal_winners.reindex(columns=x_train_cols, fill_value=0)
    x_train_medal_winners = x_train_medal_winners.reindex(columns=x_train_cols, fill_value=0)

    y_predict = np.zeros(len(x_test_reg))

    if len(x_test_medal_winners) > 0 and len(x_train_medal_winners) > 0:
        # 4. Regression: Predict medal count on the subset
        best_reg.fit(x_train_medal_winners.to_numpy(), y_train_medal_winners.to_numpy())
        y_predict_medal_winners = best_reg.predict(x_test_medal_winners.to_numpy())
        y_predict_medal_winners[y_predict_medal_winners < 0] = 0
        
        # 5. Combine predictions
        y_predict[y_predict_bool == 1] = y_predict_medal_winners
    else:
        print("Warning: Insufficient data for regression phase, predicting 0 medals.")
        
    # Evaluate
    performance = avg_std_loss(y_test_reg.to_numpy(), y_predict)
    print(f"Final Test RMSE: {performance:.2f}")
    
    # Plotting final results
    plot(y_test_reg.to_numpy(), y_predict, ttl=f"Final Test Prediction (RMSE: {performance:.2f})", line=True)
    
    return y_test_reg.to_numpy(), y_predict

# --- Main Execution ---
def run_model_pipeline():
    print("Starting Olympic Medal Prediction Pipeline (No Socio-Economic Features)...")
    
    # 1. Load Data
    try:
        raw_data = pd.read_csv('athlete_events.csv')
    except FileNotFoundError:
        print("Error: athlete_events.csv not found. Please ensure the file is in the same directory.")
        return
    
    # 2. Clean and Feature Engineer
    print("\n1. Data Cleaning and Feature Engineering (CSV features only)...")
    # Using .copy() in clean_and_feature_engineer to prevent SettingWithCopyWarning
    data = clean_and_feature_engineer(raw_data) 
    
    # 3. Data Split (Training: up to 2012, Validation: 2012, Test: 2016)
    print("2. Splitting Data...")
    x_train_reg, y_train_reg, x_valid_reg, y_valid_reg, x_test_reg, y_test_reg = train_test_split(data, validation_year)
    
    # Use classification arrays for classification tuning
    y_train_clf, y_valid_clf, y_test_clf = to_clf_data(y_train_reg, y_valid_reg, y_test_reg)
    
    # 4. Model Training and Tuning (on training/validation split)
    
    # Regression
    print("\n3. Training Regressors (Predicting Medal Count)...")
    reg_performance, reg_tuned = train_regressors(x_train_reg.to_numpy(), y_train_reg.to_numpy(), x_valid_reg.to_numpy(), y_valid_reg.to_numpy())
    
    # Classification
    print("\n4. Training Classifiers (Predicting Medal/No-Medal)...")
    # MLP (Multi-layer Perceptron) is included in the list, but only the base model is used since it has no tuning parameters specified.
    clf_performance, clf_tuned = train_classifiers(x_train_reg.to_numpy(), y_train_clf, x_valid_reg.to_numpy(), y_valid_clf)
    
    # 5. Model Selection and Plotting
    
    # Regression
    best_regressor = min(reg_performance, key=lambda k: reg_performance[k][0])
    print(f"\nBest Regressor on Validation (2012) set: {best_regressor} (RMSE: {reg_performance[best_regressor][0]:.2f})")
    plot_reg(reg_performance)
    
    # Classification
    best_classifier = max(clf_performance, key=clf_performance.get)
    print(f"Best Classifier on Validation (2012) set: {best_classifier} (Accuracy: {clf_performance[best_classifier]:.2f})")
    plot_clf(clf_performance)
    
    # 6. Final Test Prediction (2016 data)
    
    # Get the final best model estimators
    best_reg_estimator = reg_tuned[best_regressor].best_estimator_ if 'CV' in str(type(reg_tuned[best_regressor])) else reg_tuned[best_regressor]
    best_clf_estimator = clf_tuned[best_classifier].best_estimator_ if 'CV' in str(type(clf_tuned[best_classifier])) else clf_tuned[best_classifier]

    # Run the Classification-then-Regression pipeline
    predict_test_set(x_train_reg, y_train_reg, x_test_reg, y_test_reg, 
                     best_clf=best_clf_estimator, 
                     best_reg=best_reg_estimator)

if __name__ == '__main__':
    run_model_pipeline()