# Diabetes Health Indicators Dataset Analysis

## Overview
This repository contains analysis and predictive modeling based on the Diabetes Health Indicators Dataset, on Kaggle. The dataset was created by Alex Teboul. It was derived from the Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset, also on Kaggle. The objective of this analysis is to identify factors that may contribute to the diagnosis of diabetes and to build a model to predict the likelihood that an individual has diabetes given his or her health indicators data.

## Dataset Source
The dataset used in this analysis was obtained from Kaggle, under the URL:
[https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv)

Alex Teboul prepared and cleaned the dataset from the BRFSS 2015 survey responses, with an aim to provide a resource for predicting diabetes. The BRFSS survey collected data on health-related risk behaviors, chronic health conditions, and use of preventative services from over 400,000 Americans.

## Data Description
The dataset comes in three different versions, all derived from the BRFSS 2015 data:

1. `diabetes_012_health_indicators_BRFSS2015.csv` - Three classes for diabetes status.
2. `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` - A balanced dataset with a 50-50 split for respondents with/without diabetes.
3. `diabetes_binary_health_indicators_BRFSS2015.csv` - A cleaned but imbalanced dataset with two classes representing diabetes status.

The analysis in this repository is based on the third file, where the target variable `Diabetes_binary` indicates the absence (0) or presence (1) of prediabetes or diabetes.

## Criteria for Diabetes Labeling
The labeling of a respondent as having type 2 diabetes is based on the criteria set by Zidian Xie et al. in their research, "Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques". According to this study, a respondent was considered to have type 2 diabetes if they were older than 30 years, not pregnant, and answered "yes" to the question "Have you ever been told you have diabetes?"

## Exclusion Criteria
Consistent with the approach taken by Zidian Xie et al., respondents under the age of 30 were excluded from the analysis to reduce the likelihood of including type 1 diabetes cases, which have different etiological factors. Pregnant respondents and those diagnosed with prediabetes were also excluded to focus specifically on type 2 diabetes.

## Acknowledgements
I acknowledge Alex Teboul for the preparation of the dataset and making it available on Kaggle, enabling this analysis. Additionally, I acknowledge the work by Zidian Xie and his colleagues.

The BRFSS 2015 dataset can be found [here](https://www.kaggle.com/cdc/behavioral-risk-factor-surveillance-system), and Alex Teboul's notebook for data cleaning can be accessed [here](https://www.kaggle.com/alexteboul/diabetes-health-indicators-dataset-notebook).

The research by Zidian Xie et al. that inspired this dataset's creation and the exploration of the BRFSS in general can be found on the CDC's website:
[Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques](https://www.cdc.gov/pcd/issues/2019/19_0109.htm).

# Project Structure and Files Description

This project is organized into a set of folders and files to facilitate the full machine learning workflow, from data analysis to model deployment. Below is the folder structure and description of each element in this repository:

├── data

│ ├── data_dictionary.csv

│ └── diabetes_binary_health_indicators_BRFSS2015.csv

├── model

│ ├── best_model.pkl

│ └── sorted_results.pkl

├── Dockerfile

├── notebook.ipynb

├── Pipfile

├── Pipfile.lock

├── predict_test.py

├── predict.py

├── README.md

└── train.py

### Folders and Files

- `data/`: Contains the dataset and a data dictionary.
  - `data_dictionary.csv`: A file providing detailed descriptions of the dataset variables, including type and source.
  - `diabetes_binary_health_indicators_BRFSS2015.csv`: The dataset downloaded from Kaggle used for the analysis.

- `model/`: Stores the machine learning models and related information.
  - `best_model.pkl`: The serialized version of the best-performing model as determined by experimentation.
  - `sorted_results.pkl`: A DataFrame containing the results and metrics from different models tested during the experimentation phase.

### Individual Files

- `Dockerfile`: Instructions for Docker to build the image used for creating containers that run the prediction service.

- `notebook.ipynb`: A Jupyter notebook for data preprocessing, basic exploratory data analysis (EDA), and model experimentation.

- `Pipfile` and `Pipfile.lock`: These are for replicating the development and production environments including within Docker containers.

- `predict_test.py`: A Python script to test the prediction service within a container, ensuring that it functions correctly.

- `predict.py`: Contains the prediction service code. This script will be run within a Docker container in the production environment.

- `README.md`: The current file, which provides an overview and documentation for the project.

- `train.py`: The training script used to fit the final machine learning model using details (selected features, hyperparameters, etc) of a model selected from candidates created during the experimentation phase. This script can be rerun for model updates or retraining purposes.

# Workflow Sequence and Code Walkthrough

The project workflow encompasses several stages, from data preparation to feature selection, to model training, and evaluation. Below is an outline of the sequence of activities.

### Data Preparation

1. **Load and Preprocess Data**: 
   - The raw dataset is loaded and preliminary checks for missing values and duplicates are conducted.
   - Respondents under the age of 30 are excluded to focus on type 2 diabetes.

### Exploratory Data Analysis (EDA)

2. **Data Splitting**:
   - Data is split into training, validation, and test sets to ensure proper evaluation.
   - The validation and test datasets are designed to contain over 20,000 rows each for robust testing.

3. **Preprocessing**:
   - A `ColumnTransformer` is used to apply appropriate preprocessing to different types of features (nominal, ordinal, numerical). Only scaling is done as the data was already encoded.

### Feature Selection and Model Training

4. **Feature Selection**:
   - Recursive Feature Elimination with Cross-Validation (RFECV) is employed to identify the most predictive features for each model.
   
5. **Model Training and Hyperparameter Tuning**:
   - Several machine learning algorithms are explored, including Logistic Regression, Random Forest, and XGBoost.
   - Class balancing techniques are used to handle imbalanced data, with strategies like class weighting, undersampling, and SMOTE.
   - Grid search with cross-validation is conducted to find the optimal set of hyperparameters for each model.

### Model Evaluation and Selection

6. **Evaluation**:
   - Models are evaluated on the validation set using metrics like accuracy, precision, recall, F1 score, and AUC-ROC.
   - Confusion matrices are generated to assess model performance in more detail.

7. **Compilation of Results**:
   - Evaluation results are compiled into a DataFrame, which is then sorted based on F1 score and other metrics to aid in model selection.

8. **Model Selection**:
   - The 'best' model is selected based on the evaluation criteria, which in this instance, uses F1 score and a leaning towards better positive class recall than that of the negative class, while maintaining some semblance of balance between the two.

### Finalization and Deployment

9. **Final Model Testing**:
   - The selected model is fit on the combined training and validation set and evaluated on the test set to verify generalization and performance.

10. **Model Serialization**:
   - The best model is serialized and saved in the `model` directory, ready for deployment in a production environment.

# How to Replicate This Project

To replicate this project and run the code on your own system, please follow the instructions below. These steps will guide you through the process of setting up the environment, cloning the repository, and running the model training and prediction service. _A [video](https://youtu.be/RqFJBDcS-v0) of the steps below is available on my YouTube channel._

### Prerequisites

- Ensure that you have Python, pip, and Git installed on your system.

### Environment Setup

1. **Install pipenv**:
   - Pipenv is a packaging tool for Python that simplifies dependency management. Install it using the command:
     ```
     pip install pipenv
     ```
   - For more detailed instructions on pipenv installation, refer to the [official documentation](https://pipenv.pypa.io/en/latest/installation/#make-sure-you-have-python-and-pip).

2. **Clone the Repository**:
   - Create a directory for the project and navigate to it in your terminal.
   - Clone the repository with the following command:
     ```
     git clone https://github.com/wongcheefah/mlzoomcamp_midterm.git
     ```

3. **Replicate the Environment**:
   - Inside the project directory, run the following command to replicate the development environment:
     ```
     pipenv install --dev
     ```
   - This will create a virtual environment and install all the necessary dependencies as specified in the `Pipfile`.

4. **Activate the Virtual Environment**:
   - To activate the virtual environment, use the command:
     ```
     pipenv shell
     ```

### Running the Model Training Script

- To train the model using the provided script, execute:
    ```
    python3 train.py
    ```

### Building and Running the Docker Image

1. **Build the Docker Image**:
 - With Docker installed on your system, build the image using:
   ```
   docker build -t predict_diabetes .
   ```

2. **Start the Prediction Service Container**:
 - To start a Docker container that will run the prediction service, use:
   ```
   docker run -it --rm -p 9696:9696 predict_diabetes
   ```
 - The container will listen for prediction requests on port `9696`.

### Testing the Prediction Service

- Open another terminal and run the following command to test the prediction service:
    ```
    python3 predict_test.py
    ```

- The test script will send a prediction request to the service running in the Docker container. The output will display the details of the request, the prediction response, and a comparison with the actual value.

By following these steps, you should be able to replicate the project environment and run the model training and prediction service as detailed in this project.

# Conclusion

The Diabetes Health Indicators Dataset analysis and predictive modeling project aims to provide insights into the factors contributing to diabetes and offers a predictive model to assess the likelihood of diabetes in individuals. This project encapsulates the end-to-end process of a machine learning workflow, including data cleaning, exploratory data analysis, feature selection, model training and tuning, and deployment of a prediction service.

By following the replication instructions, users can set up their environment, run the model training script, and deploy the prediction service to make their own diabetes predictions based on health indicators. The provided Dockerfile ensures that the prediction service is containerized, making the deployment consistent and scalable across different platforms.

This repository is structured to not only serve as a platform for diabetes prediction but also as an educational resource for those looking to learn and apply machine learning operations (MLOps) practices. It demonstrates the use of various tools and technologies such as Jupyter notebooks for experimentation, Pipenv for environment management, and Docker for containerization.

The project underscores the importance of machine learning in healthcare analytics and the potential for predictive models to assist in early diagnosis and intervention strategies. We hope that this repository serves as a valuable resource for researchers, data scientists, and healthcare professionals interested in advancing the application of machine learning in public health.

Thank you for your interest in this project. For questions, suggestions, or contributions, please reach out through the project's GitHub repository.
