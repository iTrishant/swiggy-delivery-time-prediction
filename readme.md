# Swiggy Delivery Time Prediction
Stacked Linear Regressor combining LightGBM and Random Forest to predict food delivery time using a real-world swiggy dataset.

## Problem Statement 
In the highly competitive food delivery market, delivery time accuracy is a critical differentiator that directly impacts customer satisfaction, operational efficiency, and bottom-line profitability. 

## Solution Overview
This project builds a machine learning pipeline to accurately predict delivery times, which can be integrated into logistics, dispatch systems, or even customer apps to set expectations. 
The dataset includes real-world factors, including rider characteristics, vehicle types, weather conditions, traffic patterns, and restaurant locations. The stacked regressor model (combining LightGBM and Random Forest) provides robust predictions that adapt to dynamic delivery conditions.

## Business Value and Impact
- Improved Customer Satisfaction: Accurate ETAs reduce uncertainty and build trust, leading to higher CSAT scores
- Increased Transparency: Proactive communication about realistic delivery times, especially during delays
- Reduced Support Burden: Fewer time-related customer inquiries and complaints
- Higher Retention Rates: Satisfied customers are more likely to become repeat customers

### Stakeholders
#### For Customers
- Reliable delivery time estimates for better planning
- Reduced wait time anxiety and improved ordering confidence
- Enhanced overall food delivery experience

#### For the Business
- Improved operational efficiency and resource utilisation
- Higher customer satisfaction and retention rates
- Better strategic decision-making with data-driven insights
- Competitive advantage in the crowded food delivery market

#### For Delivery Riders
- Better route planning and time management
- Increased earning potential through optimised deliveries
- Reduced pressure and risky driving during peak hours
- Peace of mind with realistic delivery expectations

## Model Performance ((Stacked Model: LGBM + RF with Linear Regression Meta-learner)
| Metric      | Train Set | Test Set |
|-------------|-----------|----------|
| MAE         | 2.53 min  | 3.02 min |
| R² Score    | 0.8868    | 0.8374   |

## Workflow
- **Data Cleaning**: Handling missing values, feature engineering, extracting time features
- **EDA**: Visual insights into delivery trends and feature importance
- **Modelling**: Compared RF, XGB, LGBM, SVM, GB, and KNN using Optuna
- **Stacked Ensemble**: Final model combines the best RF and LGBM with Linear Regression as meta-learner
- **Evaluation Metrics**: MAE and R²

## Project Structure
<pre> 
swiggy-delivery-time-prediction/
│
├── data/
│   ├── raw/
│   │   └── swiggy.csv
│   └── processed/
│       ├── swiggy_cleaned.csv
│       └── swiggy_cleaned_final.csv
│
├── notebooks/
│   ├── 01_data_cleaning_and_inferences.ipynb
│   ├── 02_EDA_and_feature_engg.ipynb
│   └── 03_Delivery_Time_Model_Selection_and_Stacking.ipynb
│
├── models/
│   ├── best_rf.pkl
│   ├── best_lgbm.pkl
│   ├── final_stacked_model.pkl
│   ├── preprocessing_pipeline.pkl
│   ├── power_transformer.pkl
│   ├── optuna_study.pkl
│   ├── X_train_trans.pkl
│   └── y_train_pt.pkl
│
├── src/
│   ├── __init__.py
│   └── data_clean_utils.py
│
├── .gitignore
├── README.md
└── requirements.txt
</pre>

## Dataset Overview

The dataset contains food delivery records from a food delivery platform (assumed Swiggy-style), with a total of:

- **Rows**: 45,502
- **Columns**: 27 (before cleaning), 16 (after feature engineering)

### Key Columns

| Column Name                                | Description                                                      |
|--------------------------------------------|------------------------------------------------------------------|
| `age`                                      | Age of the delivery person                                       |
| `ratings`                                  | Customer ratings of the delivery person                          |
| `restaurant_latitude`, `delivery_latitude` | Geolocation coordinates for calculating delivery distance        |
| `order_time`, `order_picked_time`          | Timestamps used to compute pickup delay                          |
| `weather`, `traffic`                       | Contextual features influencing delivery time                    |
| `multiple_deliveries`                      | Number of orders assigned at once                                |
| `time_taken (target)`                      | Target variable: actual time taken for delivery (in minutes)     |

Additional features such as **pickup delay**, **haversine distance**, **time of day**, and **is_weekend** were engineered to enhance model performance.

The dataset had missing values in columns like `multiple_deliveries`, `weather`, and `city_type`. We experimented with **dropping vs imputing** these to evaluate impact on model performance.


