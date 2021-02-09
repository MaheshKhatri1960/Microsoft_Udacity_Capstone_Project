*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Microsoft Azure Automated ML & Hyperdrive Best Model Training & Deployment for Predictions From Kaggle Heart Failure Dataset 

TO DO:* This project consists of using Microsoft Azure machine learning (ML) to make predictions from the Kaggle Heart Failure <a href="https://www.kaggle.com/andrewmvd/heart-failure-clinical-data">dataset</a>. It has 3 phases. In the 1st phase, a model is trained using Microsoft Azure's <a href="https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml">AutoML (Automated ML)</a>. In the 2nd phase, Microsoft Azure's <a href="https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters">Hyperdrive</a> package which can efficiently and speedily automate the learning using combinations of a machine learning model's hyperparameters. At this stage a comparison is made between the quality of the predictions made by both the AutoML and Hyperdrive phases. The model which returned the best predictions from between the two approaches is then deployed. After deployment, it is tested with sample data to demonstrate it's functioning.     

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide.

Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

Davide Chicco, Giuseppe Jurman: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020) - (<a href="https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5">link</a>).  

The dataset was obtained from the Kaggle link shown below.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

The dataset below is available as a Microsoft Excel csv file and has the following columns over 299 rows : 
age
anaemia	creatinine_phosphokinase
diabetes
ejection_fraction
high_blood_pressure
platelets
serum_creatinine
serum_sodium
sex
smoking
time
DEATH_EVENT

The task is to create a model for predicting mortality (indicated as value of DEATH_EVENT = 1 in the above dataset) caused by Heart Failure.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

The data is accessed from this <a href="https://www.kaggle.com/andrewmvd/heart-failure-clinical-data">Kaggle Link</a> as a Microsoft CSV file. For the AutoML phase, an Azure registered dataset called 'train_data_2' is created for used by the AutoML run. For the Hyperdrive part, the Microsoft CSV file is accessed directly using Azure's TabularDatasetFactory class.   


## Automated ML

SLIDE OUTPUT - AutoML Execution with multiple models & their hyperparameters – Figure 

*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

One of the powerful AutoML settings configured was for automatic featurization (scaling & normalizing) of the input data. 

automl_settings = {
    "enable_early_stopping" : True,
    "featurization": 'auto',
    "verbosity": logging.INFO,
}

automl_config = AutoMLConfig(
    compute_target=compute_target,
    experiment_timeout_minutes=20,
    task="classification",
    primary_metric="accuracy",
    training_data=train_data_2_ds,
    label_column_name="DEATH_EVENT",
    enable_onnx_compatible_models=True,
    n_cross_validations=5,
    debug_log = "automl_errors.log",
    **automl_settings
)

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

On all the session run days AutoML selected VotingEnsemble as the best model for the primary metric - 'Accuracy'. The VotingEnsemble model makes a prediction based on an ensemble or combination of other models. Hence, it will always perform better as compared to the prediction of a single model such as Logistic Regression. 

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning

SLIDE OUTPUT - Main Steps of train.py – Python - Logistic Regression Model – Figure

SLIDE OUTPUT - Hyperdrive Overview – ‘train.py’ Execution with multiple parameters – Figure 

*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search



### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
