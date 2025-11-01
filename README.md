# Titanic Survival Prediction — Python & R Docker Environments

This repository contains two separate Dockerized environments (Python and R) for modeling passenger survival on the Titanic dataset.

---

## Project Overview
The goal of this assignment is to:
1. Read and preprocess the Titanic dataset.
2. Train a logistic regression model to predict passenger survival.
3. Run the entire workflow in Docker containers — one using **Python**, one using **R**.

---

## 1. Download the Dataset

You can download the Titanic dataset from  
[Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)

Place the following files inside `src/data/`:<br>
train.csv<br>
test.csv

---

## 2. Running the Python Container

### Build the Docker image
```bash
docker build -t titanic-python .
```
### Run the container
```bash
docker run titanic-python
```
### Output
The script will print progress logs for data loading, model training, and accuracy.<br>
Predictions will be saved to:
src/data/predictions.csv

---

## 3. Running the R Container

### Navigate to the R folder
```bash
cd src/titanic_r
```
### Build the Docker image
```bash
docker build -t titanic-r .
```
### Run the container
```bash
docker run titanic-r
```
### Output
The script will print progress logs for data loading, model training, and accuracy.<br>
Predictions will be saved to:
src/data/predictions.csv

---

## 4. Notes and Dependencies

Python container: uses pandas and scikit-learn.<br>
R container: uses tidyverse and caret.<br>
Both containers automatically install dependencies during build.<br>
The dataset (train.csv, test.csv, etc.) must be manually placed in src/data/.

---

## 5. Example Console Output

Loading Titanic dataset...<br>
Data loaded successfully!<br>

Exploring and adjusting data...<br>
Converted 'Sex' column to numeric.<br>
Filled missing values in 'Age' and 'Fare' with mean values.<br>
Training logistic regression model...<br>
Training accuracy: 0.791<br>
Test accuracy: 0.799<br>
Model training complete.<br>

Predicting on test set...<br>
Predictions complete!<br>
Saved predictions to src/data/predictions.csv<br>

---

## 6. Author

Created by Jianong (Evelyn) Xu<br>
Northwestern University — MLDS Program

