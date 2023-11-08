import pandas as pd
import pickle
from flask import Flask, request, jsonify

best_model_file = f"./model/best_model.pkl"

with open(best_model_file, "rb") as f:
    best_model = pickle.load(f)

app = Flask("diabetes")


@app.route("/predict", methods=["POST"])
def predict():
    test_case = request.get_json()
    test_df = pd.DataFrame([test_case])
    y_pred = best_model.predict_proba(test_df)[0, 1]
    Diabetes_binary = y_pred >= 0.5

    result = {
        "Diabetes_probability": float(y_pred),
        "Diabetes_binary": bool(Diabetes_binary),
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
