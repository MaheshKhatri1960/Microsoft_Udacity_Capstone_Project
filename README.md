# Microsoft Azure Automated ML & Hyperdrive Best Model Training & Deployment for Predictions From Kaggle Heart Failure Dataset 

This project consists of using Microsoft Azure machine learning (ML) to make predictions from the Kaggle Heart Failure <a href="https://www.kaggle.com/andrewmvd/heart-failure-clinical-data">dataset</a>. It has 3 phases. 

In the 1st phase, a model is trained using Microsoft Azure's <a href="https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml">AutoML (Automated ML)</a>. 

In the 2nd phase, Microsoft Azure's <a href="https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters">Hyperdrive</a> package which can efficiently and speedily automate the learning using combinations of a machine learning model's hyperparameters is used. 

In the 3rd phase, a comparison is made between the quality of the predictions made by both the AutoML and Hyperdrive phases. The model which returned the best predictions from between the two approaches is then deployed. After deployment, it is tested with sample data to demonstrate it's functioning.     

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project Overview  ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Overview.png) 

<b><p align="center">Diagram - Microsoft Udacity Azure ML Scholarship Capstone Project Overview - 3 Main Phases</p></b>

## Project Set Up and Installation

The project set up, installation & execution involved the following steps done in multiple runs in an iterative sequential manner:

- Preparation of the notebook files - AutoML's <a href="https://github.com/MaheshKhatri1960/Microsoft_Udacity_Scholarship_Capstone_Project/blob/main/automl.ipynb
">automl.ipynb</a> and Hyperdrive's <a href="https://github.com/MaheshKhatri1960/Microsoft_Udacity_Scholarship_Capstone_Project/blob/main/hyperparameter_tuning.ipynb
">hyperparameter_tuning.ipynb</a> on the local machine.
- Preparation of the python programs - <a href="https://github.com/MaheshKhatri1960/Microsoft_Udacity_Scholarship_Capstone_Project/blob/main/train.py">train.py</a> & <a href="https://github.com/MaheshKhatri1960/Microsoft_Udacity_Scholarship_Capstone_Project/blob/main/train.py">score.py</a> on the local machine.
- Uploading the above notebook & python files and dataset source to a Github account.
- Execution using the MS Cloud based Azure ML environment's Machine Learning Studio via the Udacity Labs.
- Downloading the Github files (notebooks, python & data source link) in the virtual machine on Azure ML.
- Loading the notebooks and starting the notebooks compute cluster.
- Phase 1 - Running the AutoML run notebook and making interactive changes as required.
- Phase 2 - For Hyperdrive, testing out 'train.py' independently in the Bash Git terminal to ensure flawless execution before executing the Hyperdrive run. Running the Hyperdrive run notebook and making interactive changes as required.
- Phase 3 - Registration & Deployment of the best model either from Azure ML or Hyperdrive as a web service and testing out the same.
- Downloading the notebook & run files periodically to the virtual machine and uploading back to Github.
- Cleaning up the Azure ML resources used (web service, compute cluster & notebooks compute).
- Raising of questions & further research of suggestions given by Udacity Support staff (A Big Thank you to them) as well as other similar work done as available from the Udacity Knowledgebase.

## Dataset

### Overview of the Data used for the project 
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide.

Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

<b>Davide Chicco, Giuseppe Jurman</b>: Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making 20, 16 (2020) - (<a href="https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5">Research Paper Link</a>).  

The dataset below is available as a Microsoft Excel csv file and has the following features for 299 patients (<a href="https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5/tables/1">Below Table Source</a>): 

![Figure  - Kaggle Heart Failure Prediction Dataset Columns ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Dataset_Full.png) 

<b><p align="center">Table - Brief description of the columns of the dataset</p></b>

### Task
The task is to create a model for predicting mortality (indicated as value of DEATH_EVENT = 1 in the above dataset) caused by Heart Failure.

### Access
The data from the above research is accessed from this <a href="https://www.kaggle.com/andrewmvd/heart-failure-clinical-data">Kaggle Dataset Link</a> as a Microsoft CSV file. For the AutoML phase, an Azure registered dataset called '<b>train_data_2</b>' is created for use by the AutoML run as shown in the screenshot below. 

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project AutoML Run Dataset - ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Dataset.png)

<b><p align="center">Screenshot - AutoML Execution Run - Registered Dataset 'train_data_2'</p></b>

For the Hyperdrive part, the Microsoft CSV file is accessed directly from this Github <a href="https://raw.githubusercontent.com/MaheshKhatri1960/Udacity-Capstone-Project/master/heart_failure_clinical_records_dataset.csv">link</a> using Azure's TabularDatasetFactory class.   

## Phase 1 - Automated ML

Please find below an overview diagram of the AutoML run operations with all it's different activities. This can be seen from the notebook <a href="https://github.com/MaheshKhatri1960/Microsoft_Udacity_Scholarship_Capstone_Project/blob/main/automl.ipynb">automl.ipynb</a>.

![Figure  - Overview of AutoML Run Operations ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_1.png) 

<b><p align="center">Diagram - AutoML Execution with multiple models & their hyperparameters resulting in the best model 'best_automl_model.pkl' file</p></b>

<b>Overview of the AutoML settings and configuration used</b>

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project - Auto ML Settings & Configuration ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Settings_Configuration.png)

<b><p align="center">Notebook - automl.ipynb - Code Snapshot - AutoML Run Settings & Configuration</p></b>

These AutoML Run settings are further detailed as follows :

<b>enable_early_stopping</b> - helps to avoid wastage of resources if the performance is not improving.

<b>verbosity</b> - default value set for logging run details which helps debug errors.

<b>compute target</b> - The 'MK-1B08-CC' Compute Cluster initialized at the start of the run.

<b>task</b> - 'Classification' as it is a binary level challenge (Whether death occured - Yes / No or DEATH_EVENT = 1 /0 ).

<b>primary metric</b> - has been set to 'accuracy' whose value will be optimized by Azure ML for this run.

<b>training_data</b> - set to the registered dataset 'train_data_2' with the input data

<b>label_column_name</b> - the name of the target column which is being predicted. In this case - DEATH_EVENT.

<b>enable_onnx_compatible_models</b> - helps to create models which adhered to a cross platform <a href="https://onnx.ai/">ONNX (Open Neural Network Exchange)</a> standard.

<b>n_cross_validations</b> - the number of cross validations to do to help increase the quality of the model's preditions by preventing over-fitting of the data.

<b>debug_log</b> - the file where the Automl run errors will be logged - <a href ="https://github.com/MaheshKhatri1960/Microsoft_Udacity_Scholarship_Capstone_Project/blob/main/automl_errors.log">'automl_errors.log'</a>.

<b>Featurization</b> - One of the powerful AutoML settings configured was for automatic featurization (scaling & normalizing) of the input data. This enables automatic analysis of the input data via AutoML Run Data Guardrails which does three types of analysis (Class balancing detection, Missing feature value imputation & High cardinality feature detection) as seen in the screenshot below. As mentioned in the project improvement section below, this AutoML feature is very powerful and can be used to improve model accuracy via intelligent selection of smaller datasets.

![Figure  - AutoML Run - Powerful Analysis of Input Dataset via Data Guardrails ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Data_Guardrails.png) 

<b><p align="center">Screenshot - AutoML Run - Powerful Analysis of Input Dataset via Data Guardrails </p></b>


### AutoML Execution Run

Please find screenshots below of the `RunDetails` widget.

![Figure  - AutoML Execution RunWidget Details ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_RunDetails_Widget_Execution_1.png) 

<b><p align="center">Screenshot - Start Of AutoML Execution </p></b>

![Figure  - AutoML RunWidget Details - Execution Completed](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_RunDetails_Widget_Execution_Completed.png) 

<b><p align="center">Screenshot - End Of AutoML Execution</p></b>

### Results

On all the session run days AutoML selected <b>VotingEnsemble</b> as the best model for the primary metric - <b>'Accuracy'</b>. This model makes a prediction based on an ensemble or combination of other models. Please find the screenshot of the best model <b>VotingEnsemble</b> trained with it's other data.

![Figure  - AutoML Run Completed - Best Model ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Completed_Best_Model.png) 

<b><p align="center">Screenshot - AutoML Run Completed - Best Model</p></b>

![Figure  - AutoML Run Completed - Best Model - Details - 1](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Best_Model_Details_1.png) 

<b><p align="center">Screenshot - AutoML Run - Best Model - Additional Details - 1</p></b>

![Figure  - AutoML Run Completed - Best Model - Details - 2](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Best_Model_Details_2.png) 

<b><p align="center">Screenshot - AutoML Run - Best Model - Primary metric 'Accuracy' and other metrics</p></b>

<b> All the metrics of the best model are shown below. The last 3 lines also show the metrics returned by AutoML of the other 'non-best' models. </b>
  
![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project - AutoML - Best model metrics ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Best_Model_Metrics.png)

<b><p align="center">Notebook - automl.ipynb - Code Output Snapshot - AutoML - Best model metrics</p></b>

<b>What are the results you got with your automated ML model? </b>
  
The AutoML model chose <b>'VotingEnsemble'</b> as the best algorithm with an accuracy of <b>'0.873'</b>. 

<b> AutoML Run Best Model Parameters - 1</b>

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project - AutoML - Best model Parameters ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Best_Model_Parameters.png)
![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project - AutoML - Best model Parameters 2 ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Best_Model_Parameters-2.png)

<b><p align="center">Notebook - automl.ipynb - Code Output Snapshot - AutoML - Best model Parameters</p></b>

<b> How could you have improved it? </b>

The various AutoML settings mentioned above can be experimented with to further improve the performance. Also, as mentioned in the project improvements section below, <b>'Random Forest'</b> algorithms can be studied further to improve the performance. 

Performance Difference between AutoML's 1st model - <b>VotingEnsemble</b> and the other top models. 
  
Please look at the performance returned by the top 3 AutoML models below :

![Figure  - AutoML Run - Top 3 Executed Models ](http://www.kaytek.co.in/images/msudp3/1B12_AutoML_Run_Top_3_Models.png) 

<b><p align="center">Notebook - automl.ipynb - Code Output - AutoML Run Execution showing the performance of Top 3 models</p></b>

The accuracy of about 87% returned by 'VotingEnsemble' is over 2 - 3 % higher than the accuracy of about 84 % returned by the next two models - the XgBoostClassifier.  In all the AutoML runs, 'VotingEnsemble' was consistently giving the top performance at the same level. 

## Phase 2 - Hyperparameter Tuning

Phase 2 consists of two activities. First, independent execution of <a href="https://github.com/MaheshKhatri1960/Microsoft_Udacity_Scholarship_Capstone_Project/blob/main/train.py">train.py</a> and then the Hyperdrive run itself.

Before execution of the Hyperdrive run, 'train.py' would be executed independently in the Bash Git terminal to ensure flawless Hyperdrive execution. The diagram below shows the main steps of train.py

**train.py Single Execution** - Please find below a diagram of the main steps happening in train.py. For each step, the data is shown in blue rectangles, program operations are shown in green ellipses and the arrows denote the sequence of operations.

![Figure  - Overview of main steps of Hyperdrive program train.py ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Train_Dot_Py.png) 

<b><p align="center">Diagram - Main Steps of python program train.py using SKLearn's Logistic Regression Model</p></b>

**Program Inputs** - the Excel 'heart_failure_clinical_records_dataset.csv' file which is read into the Azure (denoted by AZ) TabularDatasetFactory object. 

**'clean_data'** is a code block for a series of data cleaning steps on the csv file. Currently, nothing is being done here. As mentioned below in the project improvement section, there is scope for adding more data cleaning operations to further improve model performance.

**'clean_data'** results in two pandas data frames - 'x' & 'y'.

A **'train test split'** operation in the ratio of 70% / 30% respectively is then applied on 'x' & 'y' to get the 'x' & 'y' training & 'x' & 'y' test sets.

The classification algorithm used is logistic regression which has two hyperparameters **'C'** - the inverse of the regularization rate used to counteract model overfitting & **'max-iter'** - the number of iterations for the model to converge. 

**Program Outputs** - There are 2 outputs. The **Accuracy** score & also the file **'model.joblib'** which contains the post-execution model parameters. These outputs are available after the program execution. 

<b>Please note</b> : Only the main operations happening in 'train.py' are shown above. Other operations e.g. adding argument parsers to the program, logging the run values, etc. are not shown. 

The screenshot below shows the Git Bash Terminal Execution of train.py which as mentioned earlier is an important necessary step to avoid expensive & time consuming Hyperdrive run errors.

![Figure  - Terminal Execution Of Hyperdrive program train.py ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Train_Dot_Py_Terminal_Execution.png) 

<b><p align="center">Screenshot - Terminal Execution of train.py with hyperparameter values of **'C'** as **0.5** & **'max-iter'** as **1000** giving an Accuracy of <b>'0.789'</b>.</p></b>

Please find below an overview diagram of the Hyperdrive run operations with all it's different activities as can be seen from the notebook <a href="https://github.com/MaheshKhatri1960/Microsoft_Udacity_Scholarship_Capstone_Project/blob/main/hyperparameter_tuning.ipynb
">hyperparameter_tuning.ipynb</a>. For each step, the data is shown in blue rectangles, program operations are shown in green ellipses and the arrows denote the sequence of operations.

![Figure  - Overview of Hyperdrive Operations ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Hyperdrive.png) 

<b><p align="center">Diagram - Hyperdrive Overview – ‘train.py’ Execution with multiple values of hyperparameters as per specified policies</p></b>


<b> What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search </b> 

The algorithm chosen for this experiment was SKLearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">Logistic Regression</a>. I chose it because of it's simplicity with reasonable performance and familiarity during the earlier part of the course. 

I chose two hyperparameters **'C'** - the inverse of the regularization rate used to counteract model overfitting within a uniform range between 0.35 to 0.90 & **'max_iter'** - Maximum number of iterations to converge as a choice of any of the 3 values - 50, 75 & 100. 

These ranges used for the both the hyperparameters search were based on earlier experiences to ensure efficient hyperparameter tuning as a balance between getting good quality results without spending too much time & resources wherever possible. 

<b>Hyperparameter Values - please note</b> : 

**'C'** - smaller values specify stronger regularization and hence a better quality of the model.

**'max_iter'** - maximum number of iterations taken for the model convergence to happen. In my range above, maximum value is 100 which is the default. It is quite possible that if the **'max_iter** value is increased beyond 100, the model could give better performance.   

In this project's earlier Hyperdrive runs, I had also tried SKLearn's <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">RandomForestClassifier</a> with a choice of 3 hyperparameters - **'n_estimators'**, **'max_depth'**' & **'min_samples_split'**. Even though the accuracy results obtained in these runs were slightly better, due to paucity of time, I could not complete the model deployment of the same. As mentioned in the Project Improvement section below, plan to try out the same in the future.

**Impact of Hyperdrive's chosen policies on Hyperparameter Tuning**

One of the biggest challenges facing ML / DL practitioners tackling problems with large numbers of input data & parameters is to find high quality models at affordable time and cost. Efficient hyperparameter tuning helps in the same as it means shorter training time, lower resource consumption, and thus lower training costs. As has been found in practice,  from the entire search space available for searching for the best solutions, only a very few combinations of possible options lead to high performance. A majority of the combinations do not work and have to be discarded. Hence, hyperparameter tuning via it's policies is so essential. For Hyperdrive, there are two main policies which help achieve the same - **Early Termination**  & **Hyperparameter Sampling**. More details of my choices for this project are provided below : 

**Early Termination - Bandit Policy** 

I chose to go for a termination policy to achieve maximum resource savings during the computational runs. I chose Bandit Policy as it was the first policy in the documentation & it promised aggressive savings as compared to the other (Median / Truncation selection) policies. It has parameters of slack_factor = 0.1, an evaluation_interval of 1 & delay_evaluation of 5. This means that  starting from Run 5 (delay_evaluation = 5), at every stage of the execution after a single (evaluation_interval = 1) run, the prediction is compared and if the same is less than (1 / 1 + 0.1 (slack_factor))  or 91% of the best performing run, then it will be terminated. This helps to reduce training time, costs as well as system resources so that the system can move on to the next iteration. 

**Hyperparameter Sampling - Random**

I chose Random sampling over the other options (Grid / Bayesian) for the following reasons :

First, it supported early termination (Not supported by Bayesian sampling) as this would help me save computational resources.  
Second, due to limited experience, I did not know the extent of resource consumption for an exhaustive search (for which Grid Sampling is more suited). 
Third, it has support for both discrete and continuous values of hyperparameters. For my runs, the hyperparameter **--C** had continuous values whereas for **--max-iter** it was  discrete.


## Hyperdrive Execution

Please find below screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![Figure  - Hyperdrive Execution RunWidget Details ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Hyperdrive_RunDetails_Widget_Execution.png) 

<b><p align="center">Screenshot - Hyperdrive RunWidget Details showing Execution details in progress - 1</p></b>

![Figure  - Hyperdrive Execution RunWidget Details - 2 ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Hyperdrive_RunDetails_Widget_Execution_2.png) 

<b><p align="center">Screenshot - Hyperdrive RunWidget Details showing Execution details in progress - 2</p></b>

![Figure  - Hyperdrive Execution Completion ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Hyperdrive_RunDetails_Widget_Execution_Completed.png) 

<b><p align="center">Screenshot - Hyperdrive RunWidget Details showing Execution completed</p></b>

### Results

![Figure  - Hyperdrive Run Completed - Best Model ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Hyperdrive_Run_Best_Model.png) 

<b><p align="center">Screenshot - Hyperdrive Run Completed - Best Model</p></b>

<b>Model results</b> - The best Hyperdrive model gave an accuracy of <b>'0.833'</b>. 

<b> Model parameters </b> - The Hyperdrive best model parameters values were **'--C'** - the inverse of the regularization rate used to counteract model overfitting = <b>0.483</b> & **'--max-iter'** - the number of iterations for the model to converge = <b>100</b>. 

<b> Methods for improvement</b> - Improvements could be done both by a wider choice of hyperparameter values as well as by choice of 'Random Forest' algorithm for Hyperdrive as mentioned in the Project Improvements section below.

## Phase 3 - Best Run Model Selection & Deployment

Please find an overview diagram of the activities carried out for selection, registration & deployment of the model below.

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project - Phase 3 Overview](http://www.kaytek.co.in/images/msudp3/1B11_Mahesh_Khatri_MSUD_Azure_ML_Scholarship_Capstone_Project_Phase_3_Overview.png)

<b><p align="center">Diagram - Phase 3 - Overview of Model Selection, Registration, Deployment & Consumption</p></b>

The above diagram gives an overview of the code steps involved in model registration, deployment & consumption.

<b> Best Model Selection </b> - As can be seen from the details above, the best AutoML Run algoritm 'VotingEnsemble' gave an accuracy of <b>0.873</b> which was higher than the Hyperdrive run accuracy of <b>'0.833'</b>. Hence, the AutoML run best model was chosen for deployment as can be seen below. 

For the model deployment, in addition to the notebook, two additional files are very important :

1st is the score.py mentioned earlier.

2nd is the AutoML run environment file <a href="https://github.com/MaheshKhatri1960/Microsoft_Udacity_Scholarship_Capstone_Project_Submission/blob/main/myenv.yml
">myenv.yml</a> containing all the operating environment software dependencies. This file is automatically created as part of the AutoML run and needs to be downloaded for use. This environment file makes it very easy to deploy the model as say compared to Hyperdrive deployment. 

The deployment code can be seen in the notebook. Deployment screenshots are shown below.

![Figure  - AutoML Run - Model Deployment - 1 ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Deployed_Model_1.png) 

<b><p align="center">Screenshot - AutoML Run - Best Model - Deployment Details - 1</p></b>

![Figure  - AutoML Run - Model Deployment - 2 ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Deployed_Model_2.png) 

<b><p align="center">Screenshot - AutoML Run - Best Model - Deployment Details - 2</p></b>

![Figure  - AutoML Run - Model Deployment - 3 ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Deployed_Model_3.png) 

<b><p align="center">Screenshot - AutoML Run - Best Model - Deployment Details - 3</p></b>

<b> Instructions on how to query the endpoint with a sample input </b> - Please refer to the code snapshot below from the notebook. 

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project Model Query ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Model_Query.png)

<b><p align="center">Notebook - automl.ipynb - Code Snapshot - Best Model Deployment Query</p></b>

As can be seen above, a sample data input (input_data_point_1 in JSON format) is sent to the deployed model. Additional code for the requests and response objects as shown is executed and the model then returns an appropriate response - whether 1 (for Yes) or 0 (For No) as the prediction for DEATH_EVENT. 

For the data point queried, the deployed web service gives a response as : prediction  "{\"result\": [1]}"

## Freeing Up Azure ML Resources

The 3 major Azure ML Resources used in the project - the <b>deployed web service - 'automl-deploy-2'</b>, the <b>AutoML / Hyperdrive compute cluster - 'MK-1B08-CC'</b> & the <b>compute - 'MK-1B08-NB-Compute'</b> used for the notebooks are deleted as shown in the screenshots below:

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project Overview  ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Deployed_Model_Web_Service_Deletion.png)

<b><p align="center">Screenshot - Deployed Model Web Service 'automl-deploy-2' Deletion</p></b>

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project Overview  ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Compute_Cluster_Deletion.png)

<b><p align="center">Screenshot - Compute Cluster 'MK-1B08-CC' Deletion</p></b>

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project Overview  ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Notebook_Compute_Deletion.png)

<b><p align="center">Screenshot - Notebooks Compute 'MK-1B08-NB-Compute' Deletion</p></b>

## Project Improvement Suggestions

1 - <b>Further Research Of Random Forest Algorithms</b> - Over the last few days, I had conducted multiple runs of both Hyperdrive & AutoML for this project. While AutoML was consistently selecting 'VotingEnsemble' as the best algorithm with roughly the same accuracy performance, in the case of Hyperdrive, in case SKlearn's RandomForestClassifier is the algorithm used, it is possible to get slightly better accuracy results. I did try out some runs of the same but due to paucity of time for the project submission, could not complete the model deployment for the same. I plan to do so in the future. Also, as per the dataset authors, Random Forest algorithms can give better predictions. Hence, this needs to be studied further.  We also need to find out as to how to improve the performance of the results of the 6 AutoML's RandomForest algorithms (Nos 2,3,4,7,11 & 13) used in the AutoML run as shown in the Notebook output below.

![Figure  - AutoML Run - Executed Models - Random Forest ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Random_Forests.png) 

<b><p align="center">Notebook - automl.ipynb - Code Output - AutoML Run Execution containing multiple Random Forest algorithms</p></b>

2 - <b> Dataset Cleaning </b> - Minimal data cleaning operations were done for this project. Some researchers have suggested that perhaps the '<b>time</b>' column should not be used as a feature for prediction because it reflects the patient's followup period with the doctor and hence has no apparent impact on the accuracy of the prediction. However, the model is perhaps inaccurately considering '<b>time</b>' as the most important feature in the best AutoML run model explanation as shown in the diagram below.

![Figure  - AutoML Run Completed - Best Model - Explanation](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Best_Model_Explanation.png) 

<b><p align="center">Screenshot - AutoML Run - Best Model - Explanation showing importance of individual model features</p></b>

As can be seen, '<b>time</b> is being shown as the most important predictor feature followed by '<b>serum_creatinine</b>' & '<b>ejection fraction</b>'. As has been mentioned in the dataset section above, the dataset creators believe that "Machine learning can predict survival of patients with heart failure from <b>serum creatinine and ejection fraction alone"</b>. Hence, except for '<b>time</b>', our AutoML run best model explanations seem to confirm the researcher's findings. Hence, it may be worthwhile to remove '<b>time</b>' from the features sent to to the model for future work and see the impact on both the accuracy metric as well as the impact on the relative importance of the '<b>serum_creatinine</b>' & '<b>ejection fraction</b>' features in model prediction. 

3 - <b>Exploring smaller dataset sizes </b> - It is remarkable that even with the current dataset small size (only 299 records) as compared to usually large dataset sizes used in machine learning models, the returned accuracy figures are consistently in the high eighties. As future research beckons creation of increasingly smarter machine learning models with small dataset sizes, it would be insightful to try and further reduce the datasize to numbers less than 299 and see the impact on the metrics. Since data preparation is one of the most tedious and error prone tasks in machine learning, reducing dataset size without much impacting model prediction accuracy would be an important area of improvement. As mentioned in the AutoML section above, the AutoML Data Guardrails facility is very powerful and can be used to improve model accuracy via intelligent selection of smaller datasets.

![Figure  - AutoML Run - Powerful Analysis of Input Dataset via Data Guardrails ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Data_Guardrails.png) 

<b><p align="center">Screenshot - AutoML Run - Powerful Analysis of Input Dataset via Data Guardrails</p></b>

4 - <b> ONNX Standards Model Deployment </b> - Even though in the AutoML run configuration, the setting to create an ONNX compatible model was enabled, the deployment of the same was not explored further due to paucity of time. This is something that should be tried out in the future to help create an AI model which complies to a cross platform ONNX standard and helps increase the life & usage of the model across platforms. 

## Screen Recording
The <a href="https://www.youtube.com/watch?v=54gDLBEDybY">Youtube link</a> is a screen recording as a sequential presentation of the screenshots of the project in action. These demonstrate the dataset, a working model, it's features & codesnapshot of a sample request sent to the endpoint and its response. This presentation has been harmonized as per some of the key project diagrams shown above - Project Overview Diagram & Phase 3 Diagram.

## Standout Suggestions - The following was attempted :

<b>Application Insights Enabled</b> - As can be seen from the screenshot below, for the deployed model, application insights has be enabled which provides rich information on the operational performance of the web service which would be useful from a ML operations engineer's point of view. This has also been shown in the Phase 3 overview diagram above.

![Figure  - AutoML Run Completed - Best Model - Deployment App Insights Enabled - ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Hyperdrive_Run_Best_Model_Deployment_Enabled_App_Insights.png) 

<b><p align="center">Screenshot - Application Insights Enabled in the deployed web service</p></b>

<b> Rich Information from Application Insights </b> - The two screenshots below show the deployed model's application insights by visiting the URL shown above.

![Figure  - AutoML Run Completed - Best Model - Deployment App Insights - 1 ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Best_Model_Deployment_App_Insights_1.png) 

<b><p align="center">Screenshot - Deployed web service - Application Insights - 1 </p></b>

![Figure  - AutoML Run Completed - Best Model - Deployment App Insights - 2 ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Best_Model_Deployment_App_Insights_2.png) 

<b><p align="center">Screenshot - Deployed web service - Application Insights - 2 </p></b>

The <a href="https://www.youtube.com/watch?v=54gDLBEDybY">Youtube link</a> screen recording also shows the Application Insights screen.

<b> Thank you </b> - For your time and attention on this project.
