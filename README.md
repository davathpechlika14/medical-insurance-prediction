# Medical Insurance Cost Prediction

## Project Description
This is a Machine Learning web application built using **Streamlit**.  
It predicts the estimated **medical insurance cost** based on user input.

---

## Features
- User input from sidebar
- Real-time insurance cost prediction
- Exploratory Data Analysis (EDA)
- User feedback section
- Clean and interactive UI

---

## Machine Learning
- Model: Linear Regression (or your model)
- Dataset: Insurance dataset (CSV)
- Model saved using **pickle (.pkl)**

---

## Input Features
- Age
- Sex
- BMI
- Number of Children
- Smoking Status
- Region

---

## Project Structure
- app.py # Streamlit app
- train_model.py # Model training script
- model.pkl # Trained model
- insurance.csv # Dataset
- insurance.jpeg # Image used in UI
- requirements.txt # Dependencies

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
