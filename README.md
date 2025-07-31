# CodeAlpha_Car-Price-Prediction-with-Machine-Learning

#### Overview 
This repository hosts a machine learning project focused on predicting the selling price of used cars based on various features. In the dynamic automotive market, accurate price prediction is invaluable for both buyers and sellers. This project aims to build a robust regression model capable of estimating a car's selling price, thereby assisting in informed decision-making. We leverage a dataset containing various car attributes to train and evaluate our models.

#### Key Features & Libraries Used:
Data Manipulation & Analysis: pandas, numpy
Data Visualization: matplotlib.pyplot, seaborn
Machine Learning Models: sklearn.linear_model.LinearRegression, sklearn.linear_model.Lasso
Model Evaluation: sklearn.model_selection.train_test_split, sklearn.metrics (R-squared, MAE, MSE, RMSE)
Predictive System: Demonstrates how to make predictions on new data.

#### Dataset
The dataset used in this project (car.data.csv) contains historical information about various used cars. Key attributes include:
Car_Name: Name of the car model.
Year: Manufacturing year.
Selling_Price: The target variable – the price at which the car was sold (in Lakhs).
Present_Price: The current showroom price of the car (in Lakhs).
Driven_kms: Total kilometers driven by the car.
Fuel_Type: Type of fuel (Petrol, Diesel, CNG).
Selling_type: How the car was sold (Dealer, Individual).
Transmission: Transmission type (Manual, Automatic).
Owner: Number of previous owners.

The dataset contains 301 entries and 9 columns, with no missing values, ensuring a clean starting point for analysis.
Link For DataSet : https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars

Project Workflow and Problem-Solving Approach:
This project tackles the car price prediction problem using a standard machine learning regression workflow, implemented entirely in Python. The key steps involved are:

Data Loading and Initial Inspection:
We begin by loading the dataset into a Pandas DataFrame. Initial checks are performed to understand the data's structure, identify data types, and confirm the absence of missing values, ensuring data quality. We also examine the distribution of categorical features.

Encoding the Data: 
Categorical features (Fuel_Type, Selling_type, Transmission) are converted into numerical representations using manual encoding. This approach was chosen for simplicity due to the small, fixed number of categories, avoiding the creation of many new columns.

Splitting the Data:
The dataset is divided into training and testing sets (80% training, 20% testing) using a fixed random_state for reproducibility. The Car_Name column is dropped as it's a unique identifier and not directly useful as a numerical feature for prediction in this context.

Model Training (Linear Regression & Lasso Regression):
Two regression models, Linear Regression and Lasso Regression, are trained on the prepared training data. Linear Regression models linear relationships, while Lasso Regression adds regularization to prevent overfitting by shrinking less important feature coefficients.

Model Evaluation: The performance of both trained models is rigorously assessed using several key regression metrics:
R-squared: Measures the proportion of variance in car selling prices explained by the models.
Mean Absolute Error (MAE): Provides the average magnitude of errors in predictions.
Mean Squared Error (MSE): Penalizes larger errors more heavily.
Root Mean Squared Error (RMSE): Expresses the average error in the same units as the selling price.

These metrics are calculated for both the training and test sets to check for overfitting and evaluate generalization capability. Visualizations comparing actual vs. predicted prices further aid in this assessment.
Predictive System: Finally, the trained models are used to predict sales for new, hypothetical car data. This demonstrates the practical application of our models in estimating car selling prices.

By following this systematic approach, we aim to build an effective and interpretable car price prediction model that can serve as a valuable tool for both buyers and sellers.

How to Run the Project
Clone the Repository:

git clone https://github.com/YOUR_USERNAME/Car-Price-Prediction-Project.git
cd Car-Price-Prediction-Project

Install Dependencies:
It's recommended to use a virtual environment.

pip install pandas numpy matplotlib seaborn scikit-learn

Run the Jupyter Notebook:

jupyter notebook SourceCode_CarPricePrediction.ipynb

Open SourceCode_CarPricePrediction.ipynb in your browser and run all cells.

Interpretation of Metrics
R-squared: Both models show high R-squared values on both training and test sets. This indicates that a large proportion of the variance in car selling prices can be explained by our features. The slight drop from training to test is normal and suggests good generalization.

MAE, MSE, RMSE: These metrics quantify the average error. For Linear Regression, the RMSE on the test set is approximately 1.71 Lakhs. For Lasso, it's approximately 1.66 Lakhs. This means, on average, our model's predictions are off by about 1.6-1.7 Lakhs from the actual selling price.

Model Choice
Comparing Linear Regression and Lasso Regression, Lasso Regression shows slightly better performance on the test set (higher R-squared, lower MAE, MSE, RMSE). This suggests that Lasso's regularization, which helps prevent overfitting by shrinking less important feature coefficients, is beneficial for this dataset. Therefore, the Lasso Regression model is slightly preferred for this prediction task.

How to Make New Predictions
You can use the trained lass_reg_model (or lin_reg_model) to predict the selling price of a new car. Ensure the input data is a NumPy array with 7 features in the correct order: Year, Present_Price, Driven_kms, Fuel_Type (0=Petrol, 1=Diesel, 2=CNG), Selling_type (0=Dealer, 1=Individual), Transmission (0=Manual, 1=Automatic), Owner.

import numpy as np
Example new car data :  2018 model, Present Price 7.5 Lakhs, Driven 30000 kms, Petrol, Dealer, Manual, 0 owners
new_car_data = np.array([[2018, 7.5, 30000, 0, 0, 0, 0]])

Using the Lasso Regression model for prediction
predicted_price = lass_reg_model.predict(new_car_data)
print(f"Predicted Selling Price using Lasso Regression: {predicted_price[0]:.2f} Lakhs")
