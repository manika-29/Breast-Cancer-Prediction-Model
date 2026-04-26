# 🧠 Breast Cancer Prediction using Neural Network

## 📌 Project Overview

This project builds a **Neural Network model** to classify breast cancer tumors as **Malignant (0)** or **Benign (1)** using the **Breast Cancer Wisconsin Dataset** from `sklearn`.

The model is trained using **TensorFlow/Keras** and achieves good accuracy on test data.

---

## 📊 Dataset Information

* Source: `sklearn.datasets.load_breast_cancer()`
* Total Features: 30
* Target Classes:

  * **0 → Malignant (Cancerous)**
  * **1 → Benign (Non-cancerous)**

---

## ⚙️ Technologies Used

* Python 🐍
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow / Keras

---

## 🔍 Workflow

### 1. Data Loading & Exploration

* Loaded dataset using sklearn
* Converted into Pandas DataFrame
* Checked:

  * Shape
  * Null values
  * Statistical summary
  * Class distribution

---

### 2. Data Preprocessing

* Separated features (X) and labels (Y)
* Split dataset into:

  * Training set (80%)
  * Testing set (20%)
* Standardized data using `StandardScaler`

---

### 3. Model Building (Neural Network)

Architecture:

* Input Layer: 30 features
* Hidden Layer: 20 neurons (ReLU activation)
* Output Layer: 2 neurons (Sigmoid activation)

---

### 4. Model Compilation

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Metric: Accuracy

---

### 5. Model Training

* Epochs: 10
* Validation Split: 10%

---

### 6. Model Evaluation

* Evaluated on test dataset
* Achieved good accuracy (printed in output)

---

### 7. Visualization

* Plotted:

  * Training vs Validation Accuracy
  * Training vs Validation Loss

---

### 8. Prediction System

* Model predicts probability for each class
* Converted probabilities → class labels using `argmax`
* Built a system to predict for new input data

Example Output:

* **0 → Malignant**
* **1 → Benign**

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/Breast-Cancer-Prediction-Model.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:

```bash
jupyter notebook
```

---

## 📁 Project Structure

```
Breast-Cancer-Prediction-Model/
│── model.ipynb
│── README.md
│── requirements.txt
```

---

## 📈 Future Improvements

* Increase model accuracy using deep architectures
* Add confusion matrix & classification report
* Deploy model using Streamlit or Flask
* Use real-world datasets

---

## ✨ Conclusion

This project demonstrates how **Neural Networks** can be applied to medical datasets to assist in early detection of breast cancer, helping in better decision-making.

---

## 👩‍💻 Author

**Manika Suri**
