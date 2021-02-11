# Microsoft Azure Automated ML & Hyperdrive Best Model Training & Deployment for Predictions From Kaggle Heart Failure Dataset 

This project consists of using Microsoft Azure machine learning (ML) to make predictions from the Kaggle Heart Failure <a href="https://www.kaggle.com/andrewmvd/heart-failure-clinical-data">dataset</a>. It has 3 phases. In the 1st phase, a model is trained using Microsoft Azure's <a href="https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml">AutoML (Automated ML)</a>. In the 2nd phase, Microsoft Azure's <a href="https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters">Hyperdrive</a> package which can efficiently and speedily automate the learning using combinations of a machine learning model's hyperparameters is used. In the 3rd phase, a comparison is made between the quality of the predictions made by both the AutoML and Hyperdrive phases. The model which returned the best predictions from between the two approaches is then deployed. After deployment, it is tested with sample data to demonstrate it's functioning.     

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project Overview  ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Overview.png) 

Microsoft Udacity Azure ML Scholarship Capstone Project Overview - 3 Main Phases

## Project Set Up and Installation

The project set up involved the following steps done in multiple runs in an iterative sequential manner:

- Preparation of the notebook files - AutoML's automl.ipynb and Hyperdrive's hyperparameter_tuning.ipynb on the local machine.
- Preparation of the python programs - train.py & score.py on the local machine.
- Uploading the above notebook & python files and dataset source to a Github account.
- Starting the Cloud based Azure ML environment's Machine Learning Studio via the Udacity Labs.
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

### Task
The task is to create a model for predicting mortality (indicated as value of DEATH_EVENT = 1 in the above dataset) caused by Heart Failure.

### Access
The data from the above research is accessed from this <a href="https://www.kaggle.com/andrewmvd/heart-failure-clinical-data">Kaggle Dataset Link</a> as a Microsoft CSV file. For the AutoML phase, an Azure registered dataset called 'train_data_2' is created for used by the AutoML run. 

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project AutoML Run Dataset - ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Dataset.png)

AutoML Execution Run - Registered Dataset 'train_data_2'

For the Hyperdrive part, the Microsoft CSV file is accessed directly from this Github <a href="https://raw.githubusercontent.com/MaheshKhatri1960/Udacity-Capstone-Project/master/heart_failure_clinical_records_dataset.csv">link</a> using Azure's TabularDatasetFactory class.   

## Automated ML

Please find below an overview diagram of the AutoML run operations with all it's different activities. This can be seen from the notebook automl.ipynb.

![Figure  - Overview of AutoML Run Operations ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_1.png) 

 AutoML Execution with multiple models & their hyperparameters

<b>Overview of the `automl` settings and configuration used</b>

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project - Auto ML Settings & Configuration ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Settings_Configuration.png)

<b>enable_early_stopping</b> - helps to avoid wastage of resources if the performance is not improving.

<b>Featurization</b> - One of the powerful AutoML settings configured was for automatic featurization (scaling & normalizing) of the input data. This enables automatic analysis of the input data via AutoML Run Data Guardrails which does three types of analysis (Class balancing detection, Missing feature value imputation & High cardinality feature detection) as seen in the screenshot below. 

![Figure  - AutoML Run - Powerful Analysis of Input Dataset via Data Guardrails ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Data_Guardrails.png) 

AutoML Run - Powerful Analysis of Input Dataset via Data Guardrails

As mentioned in the project improvement section below, this AutoML feature is very powerful and can be used to improve model accuracy via intelligent selection of smaller datasets.

<b>enable_early_stopping</b> - helps to avoid wastage of resources if the performance is not improving.

<b>verbosity</b> - default value set for logging run details which helps debug errors.

<b>compute target</b> - The 'MK-1B08-CC' Compute Cluster initialized at the start of the run.

<b>task</b> - 'Classification' as it is a binary level challenge (Whether death occured - Yes / No or DEATH_EVENT = 1 /0 ).

<b>primary metric</b> - has been set to 'accuracy' whose value will be optimized by Azure ML for this run.

<b>training_data</b> - set to the registered dataset 'train_data_2' with the input data

<b>label_column_name</b> - the name of the target column which is being predicted. In this case - DEATH_EVENT.

<b>enable_onnx_compatible_models</b> - helps to create models which adhered to a cross platform <a href="https://onnx.ai/">ONNX (Open Neural Network Exchange)</a> standard.

<b>n_cross_validations</b> - the number of cross validations to do to help increase the quality of the model's preditions by preventing over-fitting of the data.

<b>debug_log</b> - the name of the file where the Automl run errors will be logged

### AutoML Execution Run

Please find screenshots below of the `RunDetails` widget.

![Figure  - AutoML Execution RunWidget Details ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_RunDetails_Widget_Execution_1.png) 

Start Of AutoML Execution 

![Figure  - AutoML RunWidget Details - Execution Completed](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_RunDetails_Widget_Execution_Completed.png) 

End Of AutoML Execution 

### Results

On all the session run days AutoML selected VotingEnsemble as the best model for the primary metric - 'Accuracy'. The VotingEnsemble model makes a prediction based on an ensemble or combination of other models. Hence, it will always perform better as compared to the prediction of a single model such as Logistic Regression. Please find the screenshot of the best model trained with it's parameters.

![Figure  - AutoML Run Completed - Best Model ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Completed_Best_Model.png) 

AutoML Run Completed - Best Model 

![Figure  - AutoML Run Completed - Best Model - Details - 1](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Best_Model_Details_1.png) 

AutoML Run - Best Model - Details 1  

![Figure  - AutoML Run Completed - Best Model - Details - 2](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Best_Model_Details_2.png) 

AutoML Run - Best Model - Details 2  

*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?


## Phase 2 - Hyperparameter Tuning

Before execution of the Hyperdrive run, 'train.py' would be executed independently in the Bash Git terminal to ensure flawless Hyperdrive execution. The diagram below shows the main steps of train.py

**train.py Single Execution** - Please find below a diagram of the main steps happening in train.py. For each step, the data is shown in blue rectangles, program operations are shown in green ellipses and the arrows denote the sequence of operations.

![Figure  - Overview of main steps of Hyperdrive program train.py ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Train_Dot_Py.png) 

Main Steps of train.py – Python - Logistic Regression Model 

**Program Inputs** - the Excel 'heart_failure_clinical_records_dataset.csv' file which is read into the Azure (denoted by AZ) TabularDatasetFactory object. 

**'clean_data'** is a series of steps on the csv file. Currently no operations are being done here. As mentioned below in the project improvement section, there is scope for adding more operations here.

**'clean_data'** results in two pandas data frames - 'x' & 'y'.

A **'train test split'** operation in the ratio of 70% / 30% respectively is then applied on 'x' & 'y' to get the 'x' & 'y' training & 'x' & 'y' test sets.

The classification algorithm used is logistic regression which has two hyperparameters **'--C'** - the inverse of the regularization rate used to counteract model overfitting & **'--max-iter'** - the number of iterations for the model to converge. 

**Program Outputs** - There are 2 outputs. The **Accuracy** score & also the file **'model.joblib'** which contains the post-execution model parameters. These outputs are available after the program execution. 

Only the main operations happening in 'train.py' are shown above. Other operations e.g. adding argument parsers to the program, logging the run values, etc. are not shown. 

The screenshot below shows the Git Bash Terminal Execution of train.py which as mentioned earlier is an important necessary step to avoid expensive & time consuming Hyperdrive run errors.

![Figure  - Terminal Execution Of Hyperdrive program train.py ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Train_Dot_Py_Terminal_Execution.png) 

Terminal Execution of train.py – Figure

Please find below an overview diagram of the Hyperdrive run operations with all it's different activities as can be seen from the notebook hyperparameter_tuning.ipynb. For each step, the data is shown in blue rectangles, program operations are shown in green ellipses and the arrows denote the sequence of operations.

![Figure  - Overview of Hyperdrive Operations ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Hyperdrive.png) 

Hyperdrive Overview – ‘train.py’ Execution with multiple parameters – Figure 

*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

**Impact of Hyperdrive's chosen policies on Hyperparameter Tuning**

One of the biggest challenges facing ML / DL practitioners tackling problems with large numbers of input data & parameters is to find high quality models at affordable time and cost. Efficient hyperparameter tuning helps in the same as it means shorter training time, lower resource consumption, and thus lower training costs. As has been found in practice,  from the entire search space available for searching for the best solutions, only a very few combinations of possible options lead to high performance. A majority of the combinations do not work and have to be discarded. Hence, hyperparameter tuning via it's policies is so essential. For Hyperdrive, there are two main policies which help achieve the same - **Early Termination**  & **Hyperparameter Sampling**. More details of my choices for this project are provided below : 

**Early Termination - Bandit Policy** 

I chose to go for a termination policy to achieve maximum resource savings during the computational runs. I chose Bandit Policy as it was the first policy in the documentation & it promised aggressive savings as compared to the other (Median / Truncation selection) policies. It has parameters of slack_factor = 0.1, an evaluation_interval of 1 & delay_evaluation of 5. This means that  starting from Run 5 (delay_evaluation = 5), at every stage of the execution after a single (evaluation_interval = 1) run, the prediction is compared and if the same is less than (1 / 1 + 0.1 (slack_factor))  or 91% of the best performing run, then it will be terminated. This helps to reduce training time, costs as well as system resources so that the system can move on to the next iteration. 

**Hyperparameter Sampling - Random**

I chose Random sampling over the other options (Grid / Bayesian) for the following reasons :

First, it supported early termination (Not supported by Bayesian sampling) as this would help me save computational resources.  
Second, due to limited experience, I did not know the extent of resource consumption for an exhaustive search (for which Grid Sampling is more suited). 
Third, it has support for both discrete and continuous values of hyperparameters. For my runs, the hyperparameter **--C** had continuous values whereas for **--max-iter** it was  discrete.


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![Figure  - Hyperdrive Execution RunWidget Details ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Hyperdrive_RunDetails_Widget_Execution.png) 

Hyperdrive RunWidget Details showing Execution details in progress – Figure 

![Figure  - Hyperdrive Execution RunWidget Details - 2 ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Hyperdrive_RunDetails_Widget_Execution_2.png) 

Hyperdrive RunWidget Details showing Execution details in progress 2 – Figure 

![Figure  - Hyperdrive Execution Completion ](http://www.kaytek.co.in/images/msudp3/1B09_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Hyperdrive_RunDetails_Widget_Execution_Completed.png) 

Hyperdrive RunWidget Details showing Execution completed – Figure 

![Figure  - Hyperdrive Run Completed - Best Model ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Hyperdrive_Run_Best_Model.png) 

Hyperdrive Run Completed - Best Model – Figure 

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

As can be seen from the details above, the best AutoML Run algoritm 'VotingEnsemble' gave an accuracy of 0.87 which was higher than the Hyperdrive run accuracy of 0.83. Hence, the AutoML run best model was chosen for deployment as can be seen below.  

![Figure  - AutoML Run - Model Deployment - 1 ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Deployed_Model_1.png) 

AutoML Run - Best Model - Deployment 1

![Figure  - AutoML Run - Model Deployment - 2 ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Deployed_Model_2.png) 

AutoML Run - Best Model - Deployment 2

![Figure  - AutoML Run - Model Deployment - 3 ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Deployed_Model_3.png) 

AutoML Run - Best Model - Deployment 3

## Freeing Up Azure ML Resources

The 3 major Azure ML Resources used in the project - deployed web service, the AutoML / Hyperdrive compute cluster & the compute used for the notebooks are deleted as shown in the screenshots below:

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project Overview  ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Deployed_Model_Web_Service_Deletion.png)

Deployed Model Web Service 'automl-deploy-2' Deletion

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project Overview  ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Compute_Cluster_Deletion.png)

Compute Cluster 'MK-1B08-CC' Deletion

![Figure  - Microsoft Udacity Azure ML Scholarship Capstone Project Overview  ](http://www.kaytek.co.in/images/msudp3/1B11_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Notebook_Compute_Deletion.png)

Notebooks Compute 'MK-1B08-NB-Compute' Deletion


## Project Improvement Suggestions
*TODO*: How to improve the project in the future

1 - <b>Further Research Of Random Forest</b> - Over the last few days, I had conducted multiple runs of both Hyperdrive & AutoML for this project. While AutoML was consistently selecting 'VotingEnsemble' as the best algorithm with roughly the same accuracy performance, in the case of Hyperdrive, in case SKlearn's RandomForestClassifier is the algorithm used, it is possible to get better accuracy results. I did try out some runs of the same but due to paucity of time for the project submission, could not complete the same. I plan to do so in the future. Also, as per the dataset authors, Random Forest algorithms can give better predictions. Hence, this needs to be studied further.  We also need to find out as to how to improve the performance of the results of the 6 AutoML's RandomForest algorithms (Nos 2,3,4,7,11 & 13) used in the AutoML run as shown in the Notebook output below.

![Figure  - AutoML Run - Executed Models - Random Forest ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Random_Forests.png) 

AutoML Run - List of Models Executed containing multiple Random Forest algorithms

2 - <b> Dataset Cleaning </b> - Minimal data cleaning operations were done for this project. Some researchers have suggested that perhaps the 'time' column should not be used as a feature for prediction because it reflects the patient's followup period with the doctor and hence has no apparent impact on the accuracy of the prediction. However, the model is considering 'time' as an important feature in the best AutoML run model explanation as shown in the diagram below.

![Figure  - AutoML Run Completed - Best Model - Explanation](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Best_Model_Explanation.png) 

AutoML Run - Best Model - Explanation 

As can be seen, 'time's is being shown as the most important predictor feature followed by 'serum_creatinine' & 'ejection fraction'. As has been mentioned in the dataset section above, the dataset creators believe that <b>"Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone"</b>. With the exception of 'time', our AutoML run best model explanations seem to confirm the researcher's findings too in terms of the importance of 'serum_creatinine' & 'ejection fraction'. Hence, it is crucial to remove 'time' from the features sent to to the model and see the impact on both the accuracy metric as well as the relative importance of 'serum_creatinine' & 'ejection fraction'. 

3 - <b>Dataset Size </b> - It is remarkable that even with the current dataset size not being very large (only 299 records) as compared to other very large dataset sizes used in machine learning models, the returned accuracy figures are consistently in the high eighties. As future research beckons creation of increasingly smarter machine learning models with small data sizes, it would be insightful to try and further reduce the datasize to numbers less than 299 and see the impact on the metrics. Since data preparation is one of the most tedious and error prone tasks in machine learning, reducing dataset size without much impacting model prediction accuracy would be an important area of improvement. As mentioned in the AutoML section above, the AutoML Data Guardrails facility is very powerful and can be used to improve model accuracy via intelligent selection of smaller datasets.

![Figure  - AutoML Run - Powerful Analysis of Input Dataset via Data Guardrails ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Data_Guardrails.png) 

AutoML Run - Powerful Analysis of Input Dataset via Data Guardrails

4 - <b> ONNX Standards Model Deployment </b> - This is something that should be tried out in the future to help create an AI model which comlpies to a cross platform ONNX standard and helps increase the life of the model.

## Screen Recording
*TODO* Provide a <a href="https://youtu.be/LekDuPgowe0">link</a> to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

1 - <b>Application Insights Emabled</b> - As can be seen from the screenshot below, for the deployed model, application insights can be enabled.

![Figure  - AutoML Run Completed - Best Model - Deployment App Insights Enabled - ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_Hyperdrive_Run_Best_Model_Deployment_Enabled_App_Insights.png) 

2 - <b> Rich Information from Application Insights </b> - The two screenshots below show the deployed model's application insights by visiting the URL shown above.

![Figure  - AutoML Run Completed - Best Model - Deployment App Insights - 1 ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Best_Model_Deployment_App_Insights_1.png) 

![Figure  - AutoML Run Completed - Best Model - Deployment App Insights - 2 ](http://www.kaytek.co.in/images/msudp3/1B10_MK_MSUD_Azure_ML_Scholarship_Capstone_Project_AutoML_Run_Best_Model_Deployment_App_Insights_2.png) 

