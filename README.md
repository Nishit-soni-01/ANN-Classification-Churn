# 📊 Customer Churn Prediction App

An end-to-end Machine Learning web application built using **Artificial Neural Networks (ANN)**, **Scikit-Learn**, and **Streamlit** to predict whether a bank customer is likely to churn (leave the bank).

---

## 🌟 Features

* **Interactive Streamlit UI**: User-friendly input sliders and dropdowns for customer demographics and account details.
* **Preprocessed Pipelines**: Automatic encoding for categorical features (`Geography`, `Gender`) and scaling via `StandardScaler`.
* **Deep Learning Model**: Trained Artificial Neural Network (`model.h5`) predicting customer churn probability in real time.
* **Visualization Ready**: Integrated with TensorBoard logging during model training.

---

## 📁 Project Structure

```text
├── app.py                   # Main Streamlit web application
├── model.h5                 # Trained Keras/TensorFlow ANN model
├── scaler.pkl               # Saved StandardScaler instance
├── label_encoder.pkl        # Saved LabelEncoder for Gender
├── onehot_encoder.pkl       # Saved OneHotEncoder for Geography
├── requirements.txt         # Project dependencies
├── logs/                    # TensorBoard training logs directory
└── README.md                # Project documentation
