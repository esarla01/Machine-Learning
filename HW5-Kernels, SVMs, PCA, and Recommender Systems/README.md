# Intro to Machine Learning 
Course: Tufts CS 135 | Fall 2023
HW5: Kernels, SVMs, PCA, and Recommender Systems  

## Overview
In HW5, we apply various machine learning techniques, including SVMs, PCA, kernel regression, and recommender systems. The homework is divided into two main parts:

1. **Conceptual Questions (50%)**: These questions cover the theoretical aspects of Support Vector Machines (SVMs), Principal Component Analysis (PCA), and Recommender Systems.
2. **Case Study on Kernel Methods for Temperature Forecasting (50%)**: You will implement kernelized regression techniques to predict temperature data using real-world datasets.

The homework will test both your understanding of key machine learning concepts and your ability to implement algorithms in Python.

## Problems
The homework is broken down into the following problems:

### Part I: Conceptual Questions (50%)
1. **Problem 0**: Conceptual questions about Support Vector Machines (SVMs).
2. **Problem 1**: Conceptual questions about Principal Component Analysis (PCA).
3. **Problem 2**: Conceptual questions about Recommender Systems.

### Part II: Case Study on Kernel Methods for Temperature Forecasting (50%)
1. **Problem 3**: Implementation of Kernel Functions (linear and squared-exponential kernels).
2. **Problem 4**: Linear Kernel + Ridge Regression.
3. **Problem 5**: Analysis of Kernel Regression Models.
4. **Problem 6**: Temperature Forecasting Model Evaluation and Hyperparameter Selection.

## Tasks

### Task 1: Implement Kernel Functions (Problem 3)
- **Objective**: Implement the linear and squared-exponential kernels as functions in Python.
- **Details**: The kernels will be used to transform data into a higher-dimensional space, enabling better model fitting for regression tasks.

### Task 2: Apply Kernel Regression (Problem 4)
- **Objective**: Use kernel methods to perform regression on the temperature dataset. Apply ridge regression with the linear kernel to predict temperature values.
- **Details**: Implement kernelized ridge regression and evaluate the performance of the model.

### Task 3: Analyze Kernel Regression Models (Problem 5)
- **Objective**: Compare and analyze different kernel regression models.
- **Details**: Investigate the impact of different kernels on prediction accuracy, and interpret the results of your models.

### Task 4: Evaluate the Temperature Forecasting Model (Problem 6)
- **Objective**: Evaluate the performance of your kernel regression model.
- **Details**: Tune hyperparameters and assess the model's predictive performance using appropriate evaluation metrics.

## Data
The dataset used for temperature forecasting is available on the course's shared resources. It includes historical temperature data that will be used for training and testing the kernel regression models.

## Report
For Part II, you will submit a report detailing the steps you took to implement the kernel methods, the models you tried, and the results you obtained. Be sure to include:
- Descriptions of the kernels used.
- A summary of hyperparameter tuning and model evaluation.
- Visualizations of model performance and comparison.

## Files to Submit
You are required to submit the following files:
1. **`hw5.py`**: Python script with all code for implementing the kernel methods, regression models, and evaluations.
2. **`hw5_report.pdf`**: Report detailing your approach, results, and analysis for Part II.
3. **`hw5_answers.pdf`**: PDF document containing answers to the conceptual questions in Part I.

