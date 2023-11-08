# Load modules
import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import pickle

# Load the data
print("Loading dataset...")
df_raw = pd.read_csv("./data/diabetes_binary_health_indicators_BRFSS2015.csv")
print("Loading completed")
print()

# Remove rows containing younger respondents (age < 30)
df = df_raw[df_raw["Age"] > 2]

# Drop duplicates to prevent data leakage
df = df.drop_duplicates()

# Load the data dictionary
print("Loading data dictionary...")
data_dict = pd.read_csv("data/data_dictionary.csv")
print("Loading completed")
print()

# Define the target variable
target_variable = "Diabetes_binary"

# Group the feature types in a dictionary
features = {}
nominals = list(data_dict[data_dict["Type"] == "Nominal"]["Variable"])
nominals.remove(target_variable)
features["nominal"] = nominals.copy()
features["ordinal"] = list(data_dict[data_dict["Type"] == "Ordinal"]["Variable"])
features["numerical"] = list(data_dict[data_dict["Type"] == "Numerical"]["Variable"])

# Separate target from features
X = df.drop(target_variable, axis=1)
y = df[target_variable]

# Set random state
random_state = 1

# For large dataset, split in the ratio [0.8, 0.1, 0.1]
print("Splitting the dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=random_state
)

# Split again to create a validation dataset. Validation and test datasets
# contain more than 20,000 rows each
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, stratify=y_test, random_state=random_state
)
print("Splitting completed")
print()

# Calculate class weight
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

# Define the base models
logistic = LogisticRegression(solver="saga", max_iter=10000, random_state=1)
random_forest = RandomForestClassifier(random_state=1)
xgboost = XGBClassifier(eval_metric="logloss", seed=1)

algorithms = [
    ("Logistic", logistic),
    ("Random Forest", random_forest),
    ("XGBoost", xgboost),
]

# Define the rebalancing strategies
balancing_strategies = {
    "Class-Weighted": "balanced",
    "Undersample": RandomUnderSampler(random_state=random_state),
    "SMOTE": SMOTE(random_state=random_state),
}

# Load the models trained in the experimentation notebook
print("Loading model hyperparameters and algorithms from experimentation ...")
print("The processing steps, parameters, algorithm and selected features of the")
print("'best' model will be used to train the final model.")
print("The definition of 'best' can change based on the desired outcome. The")
print("choice is up to the data scientist. The row index in line 111,")
print("----------------------")
print("selected_model_idx = 0")
print("----------------------")
print("can be changed to get whichever model suits the situation.")
results_file = f"./model/sorted_results.pkl"
with open(results_file, "rb") as f:
    sorted_results = pickle.load(f)
print(f"Success loading '{results_file}'.")
print()

# Select the 'best' model and get the model details
selected_model_idx = 0
selected_model = sorted_results.iloc[selected_model_idx]
strategy_name = selected_model["Balancing Strategy"]
strategy = balancing_strategies[strategy_name]
algorithm_name = selected_model["Classifier"]
algorithm = dict(algorithms)[algorithm_name]
parameters = selected_model["Parameters"]
selected_features = selected_model["Features List"]

print("Training model...")
print(f"Balancing Strategy: {strategy_name:<14}   Algorithm: {algorithm_name}")

# Add steps to the pipeline and define classifier hyperparameters
steps = []
if strategy_name != "Class-Weighted":
    # Add a balancing step
    steps.append(("balance", strategy))

# Customise preprocessor to selected features
preprocessor = ColumnTransformer(
    transformers=[
        # 'passthrough' as nominal variables are already encoded
        (
            "nominal",
            "passthrough",
            [x for x in features["nominal"] if x in selected_features],
        ),
        # No OrdinalEncoder as ordinal variables are already  encoded
        (
            "ordinal_numerical",
            StandardScaler(),
            [
                x
                for x in features["ordinal"] + features["numerical"]
                if x in selected_features
            ],
        ),
    ]
)

steps.append(("preprocessor", preprocessor))
steps.append(("classifier", algorithm))

pipeline = Pipeline(steps)

print("Pipeline built")

# Set up cross-validation
cv = StratifiedKFold(n_splits=5)

print("Cross-validation defined")

# Set parameters
param_grid = {}
for parameter, value in selected_model["Parameters"].items():
    param_grid[parameter] = [value]

print("Hyperparameters set")
print()

# Perform GridSearch
gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=3)

X_train_rfecv = X_train[selected_features]
gs.fit(X_train_rfecv, y_train)
print()
print("Training complete")
print()

# Get the best model and its selected features
best_model = gs.best_estimator_

# Evaluate the best model
X_val_rfecv = X_val[selected_features]
y_pred = best_model.predict(X_val_rfecv)
y_prob = best_model.predict_proba(X_val_rfecv)[:, 1]

# Evaluate the model
current_result = pd.DataFrame(
    [
        {
            "Balancing Strategy": strategy_name,
            "Classifier": algorithm_name,
            "Parameters": gs.best_params_,
            "Accuracy": accuracy_score(y_val, y_pred),
            "Recall": recall_score(y_val, y_pred),
            "Neg. Class Recall": recall_score(y_val, y_pred, pos_label=0),
            "Precision": precision_score(y_val, y_pred),
            "F1": f1_score(y_val, y_pred),
            "AUC-ROC": roc_auc_score(y_val, y_prob),
            "Confusion Matrix": confusion_matrix(y_val, y_pred).tolist(),
            "No. of Features": len(selected_features),
            "Features List": selected_features,
            "Model": best_model,
        }
    ]
)

print()
print("Model performance on validation data")
print(current_result.T)
print()

# Concatenate the training and validation datasets to form the final dataset
# used to train the final model
print("Concatenating test and validation data for final model training...")
X_full_train = pd.concat([X_train, X_val])
y_full_train = pd.concat([y_train, y_val])
print("Concatenation complete")
print()

print("Training final model...")
X_full_train_rfecv = X_full_train[selected_features]
gs.fit(X_full_train_rfecv, y_full_train)
print()
print("Training complete")
print()

# Get the best model and its selected features
best_model = gs.best_estimator_

# Print the classification report for the best model
print("Model performance on test data")
print("------------------------------")
print(f"Model Algorithm: {algorithm_name}")
print(f"Balancing Strategy: {strategy_name}")
print(f"Parameters: {gs.best_params_}")
print(f"Number of Features Used: {len(selected_features)}")
print(f"Features Used: {selected_features.tolist()}")

X_test_rfecv = X_test[selected_features]
y_pred = best_model.predict(X_test_rfecv)
print()
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report")
print(classification_report(y_test, y_pred))

# Save best model for use in production
best_model_file = f"./model/best_model.pkl"
with open(best_model_file, "wb") as f:
    pickle.dump(best_model, f)

print(f"The 'best model' has been saved to '{best_model_file}'.")
