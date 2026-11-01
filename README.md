# 🩺 Breast Cancer Prediction Model

An AI/ML project that uses a Neural Network to classify breast cancer data as **Malignant** or **Benign**. The project combines a trained machine learning model with a Flask web application to provide an interactive prediction interface.

## 📌 Project Overview

Breast cancer classification is an important machine learning application in healthcare research. This project uses the Wisconsin Breast Cancer dataset and a neural network built with TensorFlow/Keras to classify input data into two categories.

The Flask interface allows users to enter the required 30 numerical features and view the model's predicted class.

**Note:** This project is developed for educational purposes and is not intended for medical diagnosis or clinical decision-making.

## ✨ Features

- Neural Network-based binary classification
- Flask web application
- Interactive form for entering 30 input features
- Automatic feature scaling using a saved StandardScaler
- Prediction results displayed through a web interface
- Random sample selection for demonstration
- HTML and CSS-based user interface

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** TensorFlow, Keras, Scikit-learn
- **Data Processing:** NumPy
- **Model Persistence:** HDF5 model weights, Joblib
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Development Environment:** VS Code, Jupyter Notebook

## 📊 Dataset

The project uses Scikit-learn's built-in Wisconsin Breast Cancer dataset.

- **Input features:** 30 numerical features
- **Classification task:** Binary classification
- **Malignant:** Class 0
- **Benign:** Class 1

The model was trained using a neural network. The reported training/evaluation workflow achieved approximately **94.74% accuracy**. Actual performance may vary depending on the evaluation setup.

## 🧠 Model Workflow

1. Collect the 30 numerical input features.
2. Apply the saved StandardScaler to preprocess the input.
3. Pass the scaled features to the trained neural network.
4. Obtain the model's output scores.
5. Display the predicted class in the Flask web interface.

## 📁 Project Structure

```text
Breast-Cancer-Prediction-Model/
├── app.py
├── requirements.txt
├── breast_cancer.weights.h5
├── scaler.pkl
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/manika-29/Breast-Cancer-Prediction-Model.git
```

### 2. Navigate to the project directory

```bash
cd Breast-Cancer-Prediction-Model
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

## 🔮 Future Improvements

- Deploy the application to a cloud hosting platform.
- Improve input validation and error handling.
- Add model evaluation metrics and visualizations.
- Improve accessibility and responsive design.
- Add automated tests for the prediction pipeline.

## ⚠️ Disclaimer

This project is intended for learning and demonstration only. It is not a substitute for professional medical advice, diagnosis, or treatment. Predictions should not be used to make healthcare decisions.

## 👩‍💻 Author

**Manika Suri**

GitHub: [manika-29](https://github.com/manika-29)
