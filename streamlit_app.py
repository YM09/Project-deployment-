import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and process data
# Import Pandas module for data manipulation
import pandas as pd

# Import NumPy module for math operations
import numpy as np

# Import the randint function
from random import random
from random import randint

# Import Seaborn module for data visualization
import seaborn as sns

# Import Matplotlib module to create graphs
import matplotlib.pyplot as plt


# import machine learning model
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline

# Import the Imbalanced-Learn module to resolve class imbalance
from imblearn.over_sampling import SMOTE

# Import the Pickle module to store and retrieve Python objects
import pickle

# Import StandardScaler module for feature scaling
from sklearn.preprocessing import StandardScaler

# Import GridSearchCV and RandomizedSearchCV modules for hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score


# For split data train and split division
from sklearn.model_selection import train_test_split

# Import the ConfusionMatrixDisplay module to display the confusion matrix
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_curve, roc_auc_score, ConfusionMatrixDisplay, precision_score, f1_score, precision_recall_curve, auc

# Import the AdaBoostClassifier module for boosting
from sklearn.ensemble import AdaBoostClassifier

# to calculate the VIF value for multicollinearity checks
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Import module to ignore warnings
import warnings
# Turn off alerts
warnings.filterwarnings('ignore') # Turn off warnings

df = pd.read_csv('/content/data.csv')
df.head(10)
df.info()
df.describe()
df.isnull().sum()

# View the dataframe shape (number of rows and columns)

nRow, nCol = df. shape
print(f'\nThe data content consists of {nRow} rows and {nCol} columns')
df.columns
# Remove whitespace from all columns

df.columns = df.columns.str.strip()
# going to change the bankrupt? column to Bankrupt
df = df.rename(columns={'Bankrupt?': 'Bankrupt'})

# select columns that are not numeric columns
non_numeric_columns = df.select_dtypes(exclude=['number'])

# Check if there are columns that are not numeric columns
if not non_numeric_columns.empty:
     print("Non-numeric columns:")
     print(non_numeric_columns.columns)
else:
     print("All columns are numerical.")
# check duplicate rows in the dataset

print('{} of data is duplicated rows'.format(
     str(round(df.duplicated().sum() / df.size * 100.5))+'%'))
print('')
print('number of duplicate rows:', df.duplicated().sum())
# checking missing values
missing_values_per_column = df.isna().sum()
missing_columns = missing_values_per_column[missing_values_per_column > 0]

if missing_columns.empty:
    print("No missing value")
else:
    print("Columns with missing values: ")
    print(missing_columns)
# if the dtype is INT64 then it is considered a categorical number
numeric_columns = df.dtypes[df.dtypes != 'int64'].index
categorical_columns = df.dtypes[df.dtypes == 'int64'].index

df[categorical_columns].columns.tolist()
#Target class distribution(bankrupt)
df['Bankrupt'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, labels=['Not Bankrupt', 'Bankrupt'])
plt.title('Target Class Distribution (Bankrupt)')
plt.ylabel('') # Remove the 'Amount' label as it is not relevant for pie charts
plt.show()

# Displays value counts
print(df['Bankrupt'].value_counts())
#Distribution of the "Liability-Assests Flag"
df['Liability-Assets Flag'].value_counts().plot(kind='bar')
plt.title('Distribution Liability-Assets Flag')
plt.xlabel('Liability-Assets Flag')
plt.ylabel('Amount')
plt.show()

df['Liability-Assets Flag'].value_counts()
colors = ["Greys" , "Reds" , "Greens" , "Blues" , "Oranges" , "Purples" , "BuGn" , "BuPu" , "GnBu" , "OrRd" , "PuBu" , "PuRd" , "RdPu" , "YlGn" , "PuBuGn" , "YlGnBu"]
value = randint(0, len(colors)-1)
print(df[['Liability-Assets Flag','Bankrupt']].value_counts())
sns.countplot(x = 'Liability-Assets Flag',hue = 'Bankrupt',data = df,palette = colors[value])
#Distribution of the "net income flag"
value = randint(0, len(colors)-1)

print(df[['Net Income Flag','Bankrupt']].value_counts())
sns.countplot(x = 'Net Income Flag',hue = 'Bankrupt',data = df,palette = colors[value])
#lets find out the average profit margin based on Bankruptcy. The data based on the "bankrupt" lable and calculate the average for the profit margin feature. this will and can provid insights into how profit margins relate to the likelihood of bankruptcy.
grouped_data_profit = df.groupby('Bankrupt')[['Operating Gross Margin', 'Realized Sales Gross Margin']].mean()
grouped_data_profit.plot(kind='bar', figsize=(10, 6))
plt.title('Average Profit Margin Based on Bankruptcy')
plt.ylabel('Average Profit Margin')
plt.show()
grouped_data_profit
#correlation between liabilities and equity,
#Use of a scatter plot to see the relationship between "liability to equity"
#and "Equity to liabilty"
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Liability to Equity', y='Equity to Liability')
plt.title('Relationship between Liabilities and Equity')
plt.xlabel('Liability to Equity')
plt.ylabel('Equity to Liability')
plt.show()

# Calculates the correlation between 'Liability to Equity' and 'Equity to Liability'
correlation = df['Liability to Equity'].corr(df['Equity to Liability'])
# Prints the correlation value
print("Correlation between 'Liability to Equity' and 'Equity to Liability':", correlation)
# Analysis of the top 5 attributes that are positively and negavtively correlated with 'bankrupt'
positive_corr = df[numeric_columns].corrwith(df["Bankrupt"]).sort_values(ascending=False)[:5].index.tolist()
negative_corr = df[numeric_columns].corrwith(df["Bankrupt"]).sort_values()[:5].index.tolist()

positive_corr = df[positive_corr + ["Bankrupt"]].copy()
negative_corr = df[negative_corr + ["Bankrupt"]].copy()

print('POSITIVE CORR COLUMNS (TOP 5):')
print(positive_corr.columns)
print('\nNEGATIVE CORR COLUMNS (TOP 5):')
print(negative_corr.columns)
def corrbargraph(x_value, y_value):

    plt.figure(figsize=(15,8))
    value = randint(0, len(colors)-1)

    for i in range(1,6):
        plt.subplot(2,3,i)
        sns.barplot(x = x_value, y = y_value[i-1],data = df,palette = colors[value])

    plt.tight_layout(pad=0.5)
x_value = positive_corr.columns.tolist()[-1]
y_value = positive_corr.columns.tolist()[:-1]

corrbargraph(x_value, y_value)
x_value = negative_corr.columns.tolist()[-1]
y_value = negative_corr.columns.tolist()[:-1]

corrbargraph(x_value, y_value)
#lets find the relation between the 5 positive and negavtive correlation attribiutes to each other
# A total correlation of the top 10 attributes are given above with Bankrupt

relation = positive_corr.columns.tolist() + negative_corr.columns.tolist()
plt.figure(figsize=(8,7))
sns.heatmap(df[relation].corr(),annot=True)
data_feature = df.copy()
# Finding the top 20 of each positive correlation and negative correlation to find 40 columns that will be used as features

positive_corr_top20 = data_feature.corrwith(data_feature["Bankrupt"]).sort_values(ascending=False)[:20]
negative_corr_top20 = data_feature.corrwith(data_feature["Bankrupt"]).sort_values()[:20]

positive_corr_top20_df = positive_corr_top20.index.tolist()
negative_corr_top20_df = negative_corr_top20.index.tolist()

print('POSITIVE CORR COLUMNS (TOP 20):')
print(positive_corr_top20_df)
print(positive_corr_top20)
print('\nNEGATIVE CORR COLUMNS (TOP 20):')
print(negative_corr_top20_df)
print(negative_corr_top20)
list_of_features_40 = positive_corr_top20_df + negative_corr_top20_df

list_of_features_40
data = data_feature[list_of_features_40]
data # new dataframe with 40 columns Target and Features
#VIF checking
# Remove columns that have a VIF value above 5
def calc_vif(X):
     vif = pd.DataFrame()
     vif["variables"] = X.columns
     vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
     return vif

# Calculate VIF
X = data.iloc[:,:-1]
vif_result = calc_vif(X)

# look for variables with a VIF value > 5
high_vif_variables = vif_result[vif_result['VIF'] > 5]['variables']

# Remove variables with high VIF
df_filtered = data.drop(columns=high_vif_variables)
df_filtered
#Feature engineering
# Split data

X = df_filtered.drop('Bankrupt', axis=1)
y = df_filtered['Bankrupt']
# handling imbalance data with SMOTE
rus = SMOTE(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
# Split resampled data to training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
#model Definition
#initialize base model
# to compare 5 models with baseline parameters

models = [
    KNeighborsClassifier(),
    SVC(random_state=42),
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    AdaBoostClassifier(random_state=42)
]
#Model Training with cross Vaildation
def make_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

model_names = ['KNN', 'SVM', 'Decision Tree', 'Random Forest', 'AdaBoost']

for idx, model in enumerate(models):
    model_pipeline = make_pipeline(model)
    scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring="recall")
    print(f"------- {model_names[idx]} -------")
    print(f"Recall Score - Mean - Cross Validation: {scores.mean()}")
    print(f"Recall Score - Std - Cross Validation: {scores.std()}")
    print(f'Recall Score - Range of T-Set: {(scores.mean()-scores.std())} - {(scores.mean() + scores.std())}')
# Model training
#hyperparameter tuning.
# Create a new pipeline with Random Forest
random_forest_pipeline = make_pipeline(RandomForestClassifier(random_state=42))

# Define distribution parameters for RandomizedSearchCV
param_dist = {
     'classifier__n_estimators': [100, 200, 300],
     'classifier__max_depth': [None, 1, 10, 20],
     'classifier__min_samples_split': [2, 5, 10, 15],
     'classifier__min_samples_leaf': [1, 2, 5, 10],
     'classifier__bootstrap': [True, False]
}

# Initialize RandomizedSearchCV
random_search_rf = RandomizedSearchCV(random_forest_pipeline, param_distributions=param_dist, n_iter=50,
                                    cv=5, scoring='recall', n_jobs=-1, verbose=3, random_state=42)
random_search_rf.fit(X_train, y_train)

# Get the best parameters from RandomizedSearchCV
best_params_random_rf = random_search_rf.best_params_

print("Best RF Parameters from RandomCV:", best_params_random_rf)
random_forest_pipeline

best_rf = RandomForestClassifier(n_estimators=best_params_random_rf['classifier__n_estimators'],
                                 max_depth=best_params_random_rf['classifier__max_depth'],
                                 min_samples_split=best_params_random_rf['classifier__min_samples_split'],
                                 min_samples_leaf=best_params_random_rf['classifier__min_samples_leaf'],
                                 bootstrap=best_params_random_rf['classifier__bootstrap'],
                                 random_state=42)
best_rf
# Create a pipeline with defined steps
best_rf_pipeline = Pipeline([
     ('scaler', StandardScaler()), # Data pre-processing
     ('classifier', best_rf) # Model estimator
])

best_rf_pipeline.fit(X_train, y_train)

# Make predictions on training data
y_train_pred_rf = best_rf_pipeline.predict(X_train)

# Make predictions on test data
y_test_pred_rf = best_rf_pipeline.predict(X_test)
y_test_pred_rf
#Model Evaluation
# Evaluate model scores on the training set
recall_train_rf = recall_score(y_train, y_train_pred_rf)
pre_train_rf = precision_score(y_train, y_train_pred_rf)
f1_train_rf = f1_score(y_train, y_train_pred_rf)
print('SCORE - TRAIN SET')
print("RF Recall Score (Train):", recall_train_rf)
print("RF Precision Score (Train):", pre_train_rf)
print("RF F1 Score (Train):", f1_train_rf)


# Evaluate model scores on the test set
recall_test_rf = recall_score(y_test, y_test_pred_rf)
pre_test_rf = precision_score(y_test, y_test_pred_rf)
f1_test_rf = f1_score(y_test, y_test_pred_rf)
print('\nSCORE - TEST SET')
print("RF Recall Score (Test):", recall_test_rf)
print("RF Precision Score (test):", pre_test_rf)
print("RF F1 Score (Test):", f1_test_rf)
# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred_rf)
print(f"\nConfusion Matrix: \n{cm}")

# Classification report
print("\nClassification Report (Train):\n", classification_report(y_train, y_train_pred_rf))
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred_rf))
# Evaluate the total error on the training and test set

total_error_train_rf = len(y_train) - (len(y_train) * recall_train_rf)
error_percentage_train_rf = (total_error_train_rf / len(y_train)) * 100

print("Total Errors (Train):", total_error_train_rf)
print("Error Percentage (Train): {:.2f}%".format(error_percentage_train_rf))

# Evaluate the model on the test set
total_error_test_rf = len(y_test) - (len(y_test) * recall_test_rf)
error_percentage_test_rf = (total_error_test_rf / len(y_test)) * 100

print("\nTotal Errors (Test):", total_error_test_rf)
print("Error Percentage (Test): {:.2f}%".format(error_percentage_test_rf))
# Confusion matrix for training data

confusion_matrix_train_rf = confusion_matrix(y_train, y_train_pred_rf)
print("Confusion Matrix (Train):\n", confusion_matrix_train_rf)

# Confusion matrix for test data

confusion_matrix_test_rf = confusion_matrix(y_test, y_test_pred_rf)
print("\nConfusion Matrix (Test):\n", confusion_matrix_test_rf)

# Calculate confusion matrix for test data

confusion_matrix_test_rf = confusion_matrix(y_test, y_test_pred_rf)

# Display the confusion matrix

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_test_rf, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap='viridis')
plt.show()
# Calculating the positive predicted probability for class 1 (Bankrupt)
y_test_prob = best_rf_pipeline.predict_proba(X_test)[:, 1]

# Calculate Precision and Recall
precision, recall, _ = precision_recall_curve(y_test, y_test_prob)

# Calculate AUC for Precision-Recall curve
pr_auc = auc(recall, precision)
print(f"AUC Score for Precision-Recall Curve: {pr_auc}")

# Optional: Plotting the Precision-Recall Curve
import matplotlib.pyplot as plt
plt.figure()
plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()
# Create a function to plot Learning Curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                         n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
     plt.figure()
     plt.title(title)
     if ylim is not None:
         plt.ylim(*ylim)
     plt.xlabel("Training examples")
     plt.ylabel("Score")
     train_sizes, train_scores, test_scores = learning_curve(
         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
     train_scores_mean = np.mean(train_scores, axis=1)
     train_scores_std = np.std(train_scores, axis=1)
     test_scores_mean = np.mean(test_scores, axis=1)
     test_scores_std = np.std(test_scores, axis=1)
     plt.grid()

     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")

     plt.legend(loc="best")
     return plt

# Plot Learning Curve
title="Learning Curve (RandomForest)"
plot_learning_curve(best_rf_pipeline, title, X_train, y_train, cv=5, n_jobs=-1)
plt.show()

# Add Streamlit user inputs and visualizations here
st.title('Streamlit App')
st.write('This is a converted Streamlit app.')