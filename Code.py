import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, PoissonRegressor, LogisticRegression)
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from typing import Dict, Tuple, List, Any

validation_year: int = 2016 
prediction_year: int = 2020 

# Loss Functions
sq_loss      = lambda true_data, predict_data : 1/len(true_data) * np.sum((true_data - predict_data)**2)
avg_std_loss = lambda true_data, predict_data : np.sqrt(sq_loss(true_data, predict_data))

#Regression
class Regressor:
    def __init__(self, model_type: str = 'LinearRegression'):
        self.model_type = model_type
        
        if model_type == 'LinearReg':
            self.model = LinearRegression(fit_intercept=True)
        elif model_type == 'Ridge':
            self.model = Ridge(fit_intercept=True)
        elif model_type == 'Lasso':
            self.model = Lasso(fit_intercept=True)
        elif model_type == 'SVR':
            self.model = SVR(kernel='poly', degree=3, gamma='scale', epsilon=0.1, C=1.0, 
                             max_iter=1000000) 
        elif model_type == 'Poisson':
            self.model = PoissonRegressor(max_iter=1000000)
        elif model_type == 'RandomForest':
            self.model = RandomForestRegressor(max_depth=5, n_estimators=20)
        
        self.model_cv = None 

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model.fit(x, y)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def fit_cv(self, x: np.ndarray, y: np.ndarray):
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
            self.model_cv = GridSearchCV(estimator=self.model, param_grid=params, scoring='neg_mean_squared_error', cv=3, verbose=0).fit(x, y)
        elif self.model_type == 'LinearReg': 
            self.model.fit(x, y)
            self.model_cv = self.model

#Classifier       
class Classifier:
    def __init__(self, model_type: str = 'Logistic_Reg'):
        self.model_type = model_type
        
        # Logistic Regression is an iterative solver, keeping max_iter high.
        if model_type == 'Logistic_Reg':
            self.model = LogisticRegression(max_iter=1000000)
        elif model_type == 'SVC':
            # FIX: Added and significantly increased max_iter for the SVC iterative solver (SMO).
            self.model = SVC(kernel="rbf", C=0.025, max_iter=1000000) 
        elif model_type == 'RandomForest':
            self.model = RandomForestClassifier(max_depth=5, n_estimators=20)
        elif model_type == 'GaussianNB':
            self.model = GaussianNB()
        elif model_type == 'MLP':
            # FIX: MLPClassifier is a Neural Network, which is an iterative solver (epochs). Increased max_iter.
            self.model = MLPClassifier(alpha=1, max_iter=1000000) 

        self.model_cv = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model.fit(x, y)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def fit_cv(self, x: np.ndarray, y: np.ndarray):
        if self.model_type == 'Logistic_Reg':
            params = {'C': np.logspace(-4, 4, 20)}
            self.model_cv = GridSearchCV(estimator=self.model, param_grid=params, verbose=0, scoring='accuracy').fit(x, y)
        elif self.model_type == 'SVC':
            params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
            self.model_cv = GridSearchCV(estimator=self.model, param_grid=params, verbose=0, scoring='accuracy').fit(x, y)
        elif self.model_type == 'RandomForest':
            params = {'bootstrap': [True, False], 'max_depth': [10, 30, 50, None], 'n_estimators': [200, 400]}
            self.model_cv = GridSearchCV(estimator=self.model, param_grid=params, scoring='accuracy', cv=3, verbose=0).fit(x, y)
        elif self.model_type == 'GaussianNB':
            params = {'var_smoothing': np.logspace(0,-9, num=100)}
            self.model_cv = GridSearchCV(estimator=self.model, param_grid=params, verbose=0, scoring='accuracy').fit(x, y)
        elif self.model_type == 'MLP':
            self.model.fit(x, y)
            self.model_cv = self.model

#Graphs
def plot(x: np.ndarray, y: np.ndarray, ttl: str = None, line: bool = False, filename: str = 'final_test_prediction.png'):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel('True Medals')
    plt.ylabel('Predicted Medals')
    if line:
        lims = [min(min(x), min(y)), max(max(x), max(y))]
        plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    plt.title(ttl if ttl else 'Scatter Plot')
    plt.savefig(filename)
    plt.close()

def plot_clf_scatter(y_true: np.ndarray, y_pred: np.ndarray, best_clf: str):
    plt.figure()
    y_true_jitter = y_true + np.random.normal(0, 0.05, size=len(y_true))
    y_pred_jitter = y_pred + np.random.normal(0, 0.05, size=len(y_pred))
    
    plt.scatter(y_true_jitter, y_pred_jitter, alpha=0.5)
    plt.xticks([0, 1], ['No Medal', 'Medal'])
    plt.yticks([0, 1], ['No Medal', 'Medal'])
    plt.xlabel('True Outcome ')
    plt.ylabel('Predicted Outcome')
    plt.title(f'Classifier: {best_clf} True vs. Predicted Outcome')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('best_classifier_plot.png')
    plt.close()

def plot_clf(clf_perf: Dict[str, float], filename: str = 'classifier.png'):
    plt.figure()
    plt.bar(range(len(clf_perf)), list(clf_perf.values()), align='center')
    plt.xticks(range(len(clf_perf)), list(clf_perf.keys()), rotation=45, ha="right")
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title(f'Classifier Performance on {validation_year}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_reg(reg_perf: Dict[str, Tuple[float, float]], filename: str = 'regressor.png'):
    X_axis = np.arange(len(reg_perf))
    total_std_dev = [array[0] for array in reg_perf.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(X_axis, total_std_dev, 0.4, label = 'All Countries')
    plt.xticks(range(len(reg_perf)), list(reg_perf.keys()), rotation=45, ha="right")
    plt.ylim([0, 25])
    plt.xlabel('Regressor')
    plt.ylabel('Average Standard Deviation (RMSE)')
    plt.title(f'Regressor Performance on {validation_year}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

#Data Cleaning and Feature Engineering
def clean_and_feature_engineer(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df = df[df.Season == 'Summer'].copy()
    
    df_dedup = df.drop_duplicates(subset=['Event', 'Year', 'Team', 'Medal']).copy()
    df_dedup['Medal_Won'] = df_dedup['Medal'].apply(lambda x: 0 if pd.isna(x) else 1)
    
    athlete_count_df = df.groupby(['NOC', 'Year'])['Name'].nunique().reset_index(name='Athletes')
    medal_count_df = df_dedup.groupby(['NOC', 'Year'])['Medal_Won'].sum().reset_index(name='Medals')
    
    agg_df = pd.merge(athlete_count_df, medal_count_df, on=['NOC', 'Year'], how='outer').fillna(0)
    
    agg_df = agg_df.sort_values(by=['NOC', 'Year'])
    agg_df['Medals_Last_Games'] = agg_df.groupby('NOC')['Medals'].shift(1).fillna(0)
    
    total_medals_year = agg_df.groupby('Year')['Medals'].sum().reset_index(name='Total_Medals_Year')
    agg_df = pd.merge(agg_df, total_medals_year, on='Year', how='left')
    
    agg_df = agg_df.rename(columns={'NOC': 'Nation'})
    
    return agg_df[['Nation', 'Year', 'Athletes', 'Medals_Last_Games', 'Total_Medals_Year', 'Medals']]

#Data Splitting and Prediction Set Creation
def split_for_training_and_validation(data: pd.DataFrame, validation_year: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    training_data = data[data.Year < validation_year].copy()
    valid_data = data[data.Year == validation_year].copy()
    
    drop_cols = ['Nation', 'Medals', 'Year'] 
    
    x_train = training_data.drop(columns=drop_cols)
    y_train = training_data['Medals']
    
    x_valid = valid_data.drop(columns=drop_cols)
    y_valid = valid_data['Medals']
    
    return x_train, y_train, x_valid, y_valid

def create_2020_prediction_set(data: pd.DataFrame, prediction_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 1. Get the list of nations from the latest year
    latest_olympics = data[data.Year == data.Year.max()].copy()
    
    # 2. Create the base dataframe for test year
    predict_df = latest_olympics[['Nation']].copy().drop_duplicates()
    predict_df['Year'] = prediction_year
    
    # 3. Medals from Last Games 
    nations_last_medals = latest_olympics[['Nation', 'Medals']].rename(columns={'Medals': 'Medals_Last_Games'})
    predict_df = pd.merge(predict_df, nations_last_medals, on='Nation', how='left').fillna(0)
    
    # 4. Athlete data
    athlete_proxy = latest_olympics[['Nation', 'Athletes']].copy()
    predict_df = pd.merge(predict_df, athlete_proxy, on='Nation', how='left').fillna(0)
    
    # 5. Global Medals data
    total_medals_proxy = latest_olympics['Medals'].sum()
    predict_df['Total_Medals_Year'] = total_medals_proxy
    
    feature_cols = ['Athletes', 'Medals_Last_Games', 'Total_Medals_Year']
    x_predict = predict_df[feature_cols] 
    
    return predict_df[['Nation', 'Year']], x_predict

def to_clf_data(y_train: pd.Series, y_valid: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    yt_bool = np.zeros(len(y_train))
    yt_bool[y_train!=0] = 1
    yv_bool = np.zeros(len(y_valid))
    yv_bool[y_valid!=0] = 1
    
    return yt_bool, yv_bool

#Model Training and Prediction Functions
regressor_list = ['LinearReg','Ridge','Lasso','SVR','Poisson','RandomForest']
classifier_list = ['Logistic_Reg','SVC','GaussianNB','RandomForest', 'MLP']

def train_regressors(x_t: np.ndarray, y_t: np.ndarray, x_v: np.ndarray, y_v: np.ndarray) -> Tuple[Dict, Dict, str, np.ndarray, np.ndarray]:
    reg_performance = {}
    reg_tuned = {}
    best_regressor = ''
    best_rmse = float('inf')
    best_y_pred = None
    
    for reg_type in regressor_list:
        reg = Regressor(model_type=reg_type)
        print(f"Training Regressor: {reg_type}")
        
        reg.fit_cv(x_t, y_t)
        model_to_predict = reg.model_cv.best_estimator_ if 'CV' in str(type(reg.model_cv)) else reg.model_cv
        y_predict = model_to_predict.predict(x_v) 
        reg_tuned[reg_type] = reg.model_cv
        y_predict[y_predict<0] = 0 
        performance = avg_std_loss(y_v, y_predict)
        reg_performance[reg_type] = (performance, 0)
        print(f"RMSE: {performance:.2f}")

        if performance < best_rmse:
            best_rmse = performance
            best_regressor = reg_type
            best_y_pred = y_predict
    
    return reg_performance, reg_tuned, best_regressor, y_v, best_y_pred

def train_classifiers(x_t: np.ndarray, y_t: np.ndarray, x_v: np.ndarray, y_v: np.ndarray) -> Tuple[Dict, Dict, str, np.ndarray, np.ndarray]:
    clf_performance = {}
    clf_tuned = {}
    best_classifier = ''
    best_accuracy = float('-inf')
    best_y_pred = None
    
    for clf_type in classifier_list:
        clf = Classifier(model_type=clf_type)
        print(f"Training Classifier: {clf_type}")
        
        clf.fit_cv(x_t, y_t)
        model_to_predict = clf.model_cv.best_estimator_ if 'CV' in str(type(clf.model_cv)) else clf.model_cv
        y_predict = np.rint(model_to_predict.predict(x_v))
        clf_tuned[clf_type] = clf.model_cv
            
        performance = np.sum(y_v == y_predict) / len(y_v)
        clf_performance[clf_type] = performance
        print(f"  Accuracy: {performance:.2f}")
        
        if performance > best_accuracy:
            best_accuracy = performance
            best_classifier = clf_type
            best_y_pred = y_predict
    
    return clf_performance, clf_tuned, best_classifier, y_v, best_y_pred

def predict_2020_medals(x_train_scaled: np.ndarray, y_train_reg: pd.Series, x_valid_scaled: np.ndarray, y_valid_reg: pd.Series,
                     predict_df_info: pd.DataFrame, x_predict_scaled: np.ndarray, best_clf: Any, best_reg: Any):
    print(f"\nFinal Prediction for {prediction_year} 2020 Summer Olympics")
    
    # 1. Combine training and validation data for final model training
    x_train_final = np.vstack((x_train_scaled, x_valid_scaled))
    y_train_final = pd.concat([y_train_reg, y_valid_reg], ignore_index=True)

    # 2. Prepare final training targets 
    y_train_clf_final = np.zeros(len(y_train_final))
    y_train_clf_final[y_train_final!=0] = 1

    # 3. Classification: Predict medal/no-medal on 2020 prediction set
    best_clf.fit(x_train_final, y_train_clf_final)
    y_predict_bool = best_clf.predict(x_predict_scaled)
    
    # 4. Filter final training data for nations that won medals 
    medal_winner_indices = y_train_final[y_train_final != 0].index
    x_train_medal_winners = x_train_final[medal_winner_indices, :]
    y_train_medal_winners = y_train_final[y_train_final != 0].to_numpy()

    # 5. Filter 2020 prediction data for nations predicted to win medals
    x_predict_medal_winners = x_predict_scaled[y_predict_bool == 1, :]
    
    # Initialize final predictions to 0
    final_predictions = np.zeros(len(x_predict_scaled))

    if len(x_predict_medal_winners) > 0 and len(x_train_medal_winners) > 0:
        # 6. Regression: Predict medal count on the subset
        best_reg.fit(x_train_medal_winners, y_train_medal_winners)
        y_predict_medal_winners = best_reg.predict(x_predict_medal_winners)
        y_predict_medal_winners[y_predict_medal_winners < 0] = 0
        
        # 7. Combine predictions
        prediction_indices = np.where(y_predict_bool == 1)[0]
        final_predictions[prediction_indices] = np.rint(y_predict_medal_winners)
        
    else:
        print("Insufficient data")
        
    # 8. Create final output DataFrame
    results_df = predict_df_info.copy()
    results_df['Predicted_Medals'] = final_predictions.astype(int)
    
    # Sort and display top 10
    results_df = results_df.sort_values(by='Predicted_Medals', ascending=False)
    
    print("\nTop 10 Predicted Medal Counts for 2020:")
    print(results_df.head(10).to_string(index=False))

def model():
    print("Olympic Medal Prediction")
    try:
        raw_data = pd.read_csv('athlete_events.csv')
    except FileNotFoundError:
        print("athlete_events.csv not found")
        return
    
    print("\n1. Data Cleaning")
    data = clean_and_feature_engineer(raw_data) 

    print(f"2. Splitting Data into Training and Test Dataset")
    x_train_reg, y_train_reg, x_valid_reg, y_valid_reg = split_for_training_and_validation(data, validation_year)

    print("3.Feature Scaling")
    scaler = StandardScaler()
    
    x_train_scaled = scaler.fit_transform(x_train_reg)
    x_valid_scaled = scaler.transform(x_valid_reg)
    y_train_clf, y_valid_clf = to_clf_data(y_train_reg, y_valid_reg)
    y_train_reg_np = y_train_reg.to_numpy()
    y_valid_reg_np = y_valid_reg.to_numpy()
    
    # Regression
    print("\n4. Training Regressors on 2016 (Validation)")
    reg_performance, reg_tuned, best_regressor, y_true_reg, y_pred_reg = train_regressors(
        x_train_scaled, y_train_reg_np, x_valid_scaled, y_valid_reg_np
    )
    
    # Classification
    print("\n5. Training Classifiers on 2016 (Validation)")
    clf_performance, clf_tuned, best_classifier, y_true_clf, y_pred_clf = train_classifiers(
        x_train_scaled, y_train_clf, x_valid_scaled, y_valid_clf
    )
    
    # Comparison Plots
    print(f"\nBest Regressor on Validation ({validation_year}) set: {best_regressor} (RMSE: {reg_performance[best_regressor][0]:.2f})")
    plot_reg(reg_performance)
    
    print(f"Best Classifier on Validation ({validation_year}) set: {best_classifier} (Accuracy: {clf_performance[best_classifier]:.2f})")
    plot_clf(clf_performance)
    
    # True vs Predicted Plots
    plot(y_true_reg, y_pred_reg, 
         ttl=f'Regressor: {best_regressor} True vs. Predicted Medals (2016)', 
         line=True, 
         filename='best_regressor_plot.png') 
    
    plot_clf_scatter(y_true_clf, y_pred_clf, best_classifier)
    

    predict_df_info, x_predict_unscaled = create_2020_prediction_set(data, prediction_year)
    print("6.Creating and Scaling 2020 Prediction Set")
    x_predict_scaled = scaler.transform(x_predict_unscaled)
    
    best_reg_estimator = reg_tuned[best_regressor].best_estimator_ if 'CV' in str(type(reg_tuned[best_regressor])) else reg_tuned[best_regressor]
    best_clf_estimator = clf_tuned[best_classifier].best_estimator_ if 'CV' in str(type(clf_tuned[best_classifier])) else clf_tuned[best_classifier]
    
    predict_2020_medals(x_train_scaled, y_train_reg, x_valid_scaled, y_valid_reg, predict_df_info, x_predict_scaled, 
                     best_clf=best_clf_estimator, 
                     best_reg=best_reg_estimator)

if __name__ == '__main__':
    model()