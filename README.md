# 🌾 Crop Yield Prediction Using Machine Learning Models

This project is a Machine Learning-based system designed to predict agricultural crop yields. Analysing soil and environmental factors, it helps determine the best conditions for maximizing productivity.

---

## 📌 Project Overview

Agriculture is the backbone of the economy, but yield prediction remains a challenge due to various environmental factors. This project leverages **Supervised Machine Learning** to forecast crop yields based on:

- **Soil Parameters:** Nitrogen (N), Phosphorus (P), Potassium (K), and pH levels.
- **Weather Conditions:** Temperature, Humidity, and Rainfall.
- **Target:** Accurate yield prediction to assist in decision-making and resource management.

---

## 🚀 Features

- **Data Preprocessing:** Cleaning and handling agricultural datasets.
- **Exploratory Data Analysis (EDA):** Visualising correlations between climate patterns and crop growth.
- **Model Implementation:** Comparison of multiple algorithms (Random Forest, XGBoost, Decision Trees).
- **Performance Evaluation:** Models are assessed using R-squared ($R^2$) and Mean Absolute Error (MAE).

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Google Colab

---

## ⚙️ Local Setup & Installation Guide

Follow the steps below to set up and run this project on your own computer:

### 1. Clone the Repository

First, to download the project to your PC, enter the following command in your terminal:

```bash
git clone https://github.com/Mahabub-Jamil/Crop-Yield-Prediction-Using-Machine-Learning-Models.git
cd Crop-Yield-Prediction-Using-Machine-Learning-Models
```

### 2. Create a Virtual Environment (Recommended)

It is a good idea to create a virtual environment to keep the library versions isolated:

**Windows-এর জন্য:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux-এর জন্য:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies

To install all the required libraries for the project, run the following command:

```bash
pip install -r requirements.txt
```

### 4. Run the Project

To run the project, open Jupyter Notebook by using the following command in your terminal:

```bash
jupyter notebook
```

Then, from the browser, select your main .ipynb file and run all the cells.
---

## 📂 Project Structure

```text
├── dataset/                # CSV files containing agricultural data
├── notebooks/              # Jupyter notebooks for EDA and training
├── models/                 # Pre-trained/saved models (.pkl)
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

---

## 🤝 Contributing

Contributions are always welcome! If you want to improve the model or add new features:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

---

If you find this project useful, please consider giving it a ⭐!
