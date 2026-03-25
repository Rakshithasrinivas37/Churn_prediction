import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

## Parameters
C=1.0
n_splits = 5
output_file = f'model_C={C}.bin'

## -------------------------------Data preparation------------------------------

df = pd.read_csv('data/churn_prediction_data.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

## To convert total charges column values to numeric value
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

## To convert churn values from yes/no to 0/1
df.churn = (df.churn == 'yes').astype(int)

## Split the dataset
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
y_test = df_test['churn'].values


## Divide the dataset columns into numerical and categorical columns
numerical_columns = ['tenure', 'monthlycharges', 'totalcharges']
categorical_columns = ['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']

## ---------------------------------------Training----------------------------------------

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical_columns + numerical_columns].to_dict(orient='records')

    dv = DictVectorizer()

    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=10000) ## If C value is large, regularization is not strong, and if C is small, regularization is strong
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical_columns + numerical_columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

scores = []

print(f"Validation with C={C}")

fold = 0
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

for train_idx, val_idx in tqdm(kfold.split(df_full_train, df_full_train['churn']), total=n_splits):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train['churn'].values
    y_val = df_val['churn'].values

    dv, model = train(df_train, y_train, C)

    y_pred = predict(df_val, dv, model)

    auc_score = roc_auc_score(y_val, y_pred)
    scores.append(auc_score)

    print(f"AUC on fold {fold}: {auc_score}")
    fold += 1

print("\nValidation Results: ")
print('C=%s %.3f +- %.3f\n' % (C, np.mean(scores), np.std(scores)))

## Training the final model
print("Training the final model")
dv, model = train(df_full_train, df_full_train['churn'].values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)

print(f"AUC of final model: {auc}\n")
print("Training Completed!!!")

## ------------------------------------------Model Save-----------------------------------------

## Saving the model
with open(output_file, 'wb') as file:
    pickle.dump((dv, model), file)

print(f"Model saved to {output_file}")