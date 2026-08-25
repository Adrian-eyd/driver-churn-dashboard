# Driver Churn Prediction & Retention Analytics

## Project Overview

This project uses machine learning to predict which drivers are more likely to stop using a mobility platform.

The goal is to help the business identify drivers who may be at risk of leaving and take action early to improve driver retention.

The project includes data analysis, machine learning, model evaluation, and an interactive Streamlit dashboard.

## Business Problem

Driver churn can affect the availability and reliability of a mobility platform. When drivers leave, the business may need to spend more time and money recruiting and onboarding new drivers.

A predictive model can help identify drivers who are more likely to churn so that the business can focus retention efforts on the drivers who need the most attention.

## Objectives

* Understand the main factors related to driver churn.
* Explore patterns in driver activity and behavior.
* Build a machine learning model to predict churn.
* Evaluate the performance of the model.
* Create an interactive dashboard to communicate the results.
* Provide insights that can support driver retention decisions.

## Project Workflow

The project follows these main steps:

1. Data loading and cleaning
2. Exploratory data analysis
3. Feature preparation
4. Model training
5. Model evaluation
6. Saving the trained model
7. Building an interactive Streamlit dashboard

## Machine Learning

A Logistic Regression model was used to predict driver churn.

The model was trained using a train/test split and evaluated using several performance metrics.

### Model Metrics

| Metric    |Result |
| --------- | ------|
| Accuracy  | 75.7% |
| Precision | 89.7% |
| Recall    | 73.6% |
| ROC-AUC   | 84.9% |

## Key Insights

The analysis looks at factors that may be associated with driver churn, such as driver activity, engagement, and other available driver characteristics.

The model can be used to identify drivers with a higher predicted risk of churn.

These predictions can help the business consider targeted retention actions such as:

* Driver engagement campaigns
* Targeted incentives
* Operational support
* Improved driver communication
* Further investigation of driver experience

## Interactive Dashboard

The project includes a Streamlit dashboard that allows users to explore the data and model results.

**Live Dashboard:** Add your Streamlit link here

The dashboard provides a more interactive way to understand driver churn and explore the model's predictions.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn
* Plotly
* Jupyter Notebook

## Project Structure

```text
driver-churn-prediction/
│
├── app.py
├── train_model.py
├── churn_model.pkl
├── requirements.txt
├── README.md
└── results.csv
```

## What I Learned

This project helped me strengthen my skills in:

* Data cleaning and preparation
* Exploratory data analysis
* Feature engineering
* Machine learning
* Model evaluation
* Python
* Building interactive data applications
* Communicating analytical results to a business audience

## Future Improvements

Future versions of the project could include:

* Testing additional machine learning models
* Hyperparameter tuning
* Model explainability using SHAP
* More detailed driver segmentation
* Automated model retraining
* Additional retention recommendations

## Author

**Adrian Danso**

Data Analytics | Business Intelligence | Machine Learning

