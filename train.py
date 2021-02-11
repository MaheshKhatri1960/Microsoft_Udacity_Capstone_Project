# MK-1A26
# Reused from p1_train.py
# MK-0K23 - train_C_dot_50_mi_1000_ts_03.py copied to train.py
# MK-0K23 - https://knowledge.udacity.com/questions/386233
from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.dataset import Dataset

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

#ds = ### YOUR CODE HERE ###

print ("before ds")
datastore_path = "https://raw.githubusercontent.com/MaheshKhatri1960/Udacity-Capstone-Project/master/heart_failure_clinical_records_dataset.csv"

#ds = Dataset.Tabular.from_delimited_files(path=datastore_path, validate=True, include_path=False, infer_column_types=True, set_column_types=None, separator=',', header=True, partition_format=None, support_multi_line=False, empty_as_string=False, encoding='utf8')
#ds = Dataset.Tabular.from_delimited_files(path=datastore_path)
ds = TabularDatasetFactory.from_delimited_files(path=datastore_path)

print ("after ds")

def clean_data(data):

    x_df = data.to_pandas_dataframe().dropna()
    y_df = x_df.pop("DEATH_EVENT")
        
    return x_df,y_df

x, y = clean_data(ds)

# TODO: Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

### YOUR CODE HERE ###a

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=0.50, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=1000, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(model, './outputs/model.joblib')

if __name__ == '__main__':
    main()