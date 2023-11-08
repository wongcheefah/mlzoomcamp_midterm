import requests


url = "http://localhost:9696/predict"

test_case = {
    "HighBP": 1.0,
    "HighChol": 1.0,
    "CholCheck": 1.0,
    "BMI": 40.0,
    "Smoker": 1.0,
    "Stroke": 0.0,
    "HeartDiseaseorAttack": 0.0,
    "PhysActivity": 1.0,
    "Fruits": 1.0,
    "Veggies": 1.0,
    "HvyAlcoholConsump": 0.0,
    "AnyHealthcare": 1.0,
    "NoDocbcCost": 0.0,
    "GenHlth": 3.0,
    "MentHlth": 0.0,
    "PhysHlth": 1.0,
    "DiffWalk": 1.0,
    "Sex": 0.0,
    "Age": 10.0,
    "Education": 6.0,
    "Income": 8.0,
}

diabetes_actual = "Yes"

response = requests.post(url, json=test_case).json()
print("Case details:")
print(test_case)
print()
print("Response:")
print(response)
print()

if response["Diabetes_binary"]:
    print("Diabetes Predicted: Yes")
    print(f"Diabetes Actual:    {diabetes_actual}")
else:
    print("Diabetes Predicted: No")
    print(f"Diabetes Actual:    {diabetes_actual}")

print()
