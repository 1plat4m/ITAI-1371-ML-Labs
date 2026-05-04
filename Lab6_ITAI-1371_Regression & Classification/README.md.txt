Module 06 Lab: Regression and Classification Models
This README summarizes the work completed in Module 06 Lab - Regression and Classification Models.ipynb. The lab focused on implementing the two primary types of supervised learning: predicting continuous values and identifying discrete categories. 
Lab Overview
The objective was to distinguish between regression and classification tasks by building linear models using the Titanic dataset. 
Part 1: Linear Regression (Predicting Continuous Values)
•	Task: Predict the passenger Fare based on Age and Pclass. 
•	Implementation:
o	Cleaned the data by dropping rows with missing Age values. 
o	Trained a LinearRegression model using scikit-learn. 
•	Evaluation: Used Mean Squared Error (MSE) to measure the average squared difference between predicted and actual fares. 
o	Result: The model yielded an MSE of 3364.92 and a Root Mean Squared Error (RMSE) of 58.01. 
o	Insights: The negative coefficients for Age and Pclass indicated that as these values increase, the predicted fare tends to decrease. 
Part 2: Logistic Regression (Classification)
•	Task: Predict passenger survival (Survived) based on Age, Pclass, and Sex. 
•	Implementation:
o	Encoded the Sex column into numerical values (male: 0, female: 1). 
o	Trained a LogisticRegression model, which predicts the probability of a class label. 
•	Evaluation: Used Accuracy Score to determine the percentage of correct survival predictions. 
o	Result: The model achieved an accuracy of 74.83%. 
What I Learned
•	Supervised Learning Categorization: I solidified my understanding that regression is for continuous numerical outputs, while classification is for discrete labels. 
•	Model Attributes: I learned to interpret the .coef_ attribute, which represents the weights assigned to each feature. These values quantify the impact each feature has on the final prediction. 
•	Metric Selection: I learned why evaluation metrics must match the task. Accuracy is ineffective for regression because continuous values rarely match exactly; instead, error-size metrics like MSE are required. 
Challenges Faced
•	Interpreting Coefficients: Understanding how a one-unit increase in a feature like Pclass affects the Fare required careful analysis of the learned weights. 
•	Handling Categorical Data: Ensuring the Sex column was mapped to numbers before training was a critical preprocessing step for the Logistic Regression model to function. 
•	Regression Sensitivity: Noticing the high MSE for fare prediction highlighted that linear models may struggle with highly variable continuous data or that more features might be needed to improve accuracy. 

