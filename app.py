from pathlib import Path
import random
import joblib
import numpy as np
from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer
from tensorflow import keras


# -----------------------------------------
# 1. PROJECT CONFIGURATION
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

WEIGHTS_PATH = BASE_DIR / "breast_cancer.weights.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"

app = Flask(__name__)


# -----------------------------------------
# 2. LOAD DATASET AND FEATURE NAMES
# -----------------------------------------

dataset = load_breast_cancer()
feature_names = dataset.feature_names.tolist()

# Remember the previously loaded sample index.
# This helps prevent consecutive duplicate samples.
last_sample_index = None


# -----------------------------------------
# 3. RECREATE THE TRAINED NEURAL NETWORK
# -----------------------------------------

model = keras.Sequential([
    keras.layers.Input(shape=(30,)),
    keras.layers.Flatten(),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(2, activation="sigmoid")
])


# -----------------------------------------
# 4. LOAD TRAINED WEIGHTS AND SCALER
# -----------------------------------------

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"Model weights not found: {WEIGHTS_PATH}\n"
        "Copy breast_cancer.weights.h5 into the project folder."
    )

if not SCALER_PATH.exists():
    raise FileNotFoundError(
        f"Scaler file not found: {SCALER_PATH}\n"
        "Copy scaler.pkl into the project folder."
    )

model.load_weights(str(WEIGHTS_PATH))
scaler = joblib.load(SCALER_PATH)

print("Model weights loaded successfully!")
print("Scaler loaded successfully!")


# -----------------------------------------
# 5. HOME PAGE
# -----------------------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        feature_names=feature_names,
        values={},
        prediction=None,
        scores=None,
        error=None
    )


# -----------------------------------------
# 6. LOAD A DIFFERENT SAMPLE
# -----------------------------------------

@app.route("/sample")
def load_sample():
    global last_sample_index

    # Exclude the previously selected sample.
    available_indices = [
        i for i in range(len(dataset.data))
        if i != last_sample_index
    ]

    # Select another sample from the original dataset.
    sample_index = random.choice(available_indices)
    last_sample_index = sample_index

    sample_values = dataset.data[sample_index]

    values = {
        f"feature_{i}": str(float(value))
        for i, value in enumerate(sample_values)
    }

    return render_template(
        "index.html",
        feature_names=feature_names,
        values=values,
        prediction=None,
        scores=None,
        error=None
    )


# -----------------------------------------
# 7. PREDICTION ROUTE
# -----------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    values = {}

    try:
        input_values = []

        # Read all 30 feature values.
        for i, feature in enumerate(feature_names):
            field_name = f"feature_{i}"

            raw_value = request.form.get(
                field_name, ""
            ).strip()

            values[field_name] = raw_value

            if raw_value == "":
                raise ValueError(
                    f"Please enter a value for {feature}."
                )

            try:
                number = float(raw_value)
            except ValueError:
                raise ValueError(
                    f"Please enter a valid number for {feature}."
                )

            if not np.isfinite(number):
                raise ValueError(
                    f"Invalid number for {feature}."
                )

            input_values.append(number)

        # Convert the input into a NumPy array.
        input_array = np.asarray(
            input_values,
            dtype=np.float32
        ).reshape(1, 30)

        # Apply the same scaler used during training.
        input_scaled = scaler.transform(input_array)

        # Generate model output.
        output = model.predict(
            input_scaled,
            verbose=0
        )[0]

        # Select the class with the highest model output.
        predicted_class = int(np.argmax(output))

        # Dataset labels:
        # 0 = malignant
        # 1 = benign
        prediction = (
            "Malignant"
            if predicted_class == 0
            else "Benign"
        )

        # Raw sigmoid scores, not calibrated probabilities.
        scores = {
            "malignant": round(
                float(output[0]) * 100, 2
            ),
            "benign": round(
                float(output[1]) * 100, 2
            )
        }

        return render_template(
            "index.html",
            feature_names=feature_names,
            values=values,
            prediction=prediction,
            scores=scores,
            error=None
        )

    except (ValueError, TypeError) as e:
        return render_template(
            "index.html",
            feature_names=feature_names,
            values=values,
            prediction=None,
            scores=None,
            error=str(e)
        ), 400


# -----------------------------------------
# 8. START FLASK APPLICATION
# -----------------------------------------

if __name__ == "__main__":
    print("Starting Breast Cancer Prediction UI...")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )