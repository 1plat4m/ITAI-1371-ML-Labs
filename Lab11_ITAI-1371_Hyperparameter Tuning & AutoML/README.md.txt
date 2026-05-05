Module 07 Lab: Better Model Evaluation
This lab focused on moving beyond simple accuracy to implement more sophisticated and reliable machine learning evaluation techniques using the Titanic dataset. 
________________________________________
What I Did
In this lab, I developed a classification pipeline using scikit-learn to predict passenger survival. The project was divided into four main technical segments: 
1.	Model Training: I prepared the Titanic dataset by encoding categorical variables (Sex) and handling missing values. I then trained a Logistic Regression model using features like Age, Passenger Class, Sex, and Fare. 
2.	Confusion Matrix Visualization: I used sklearn.metrics.confusion_matrix and seaborn to create a heatmap. This allowed for a visual breakdown of: 
o	True Positives (TP) and True Negatives (TN)
o	Type I Errors (False Positives) and Type II Errors (False Negatives). 
3.	Classification Metrics: I generated a comprehensive classification_report to analyze the performance of each class (Died vs. Survived). 
4.	Robust Evaluation: I implemented 5-Fold Cross-Validation to ensure the model's performance was consistent across different subsets of the data, rather than relying on a single potentially "lucky" split. 
________________________________________
What I Learned
Through this lab, I gained a deeper understanding of the trade-offs in model evaluation:
•	Beyond Accuracy: I learned that accuracy can be misleading, and metrics like Precision (correctness of positive predictions) and Recall (ability to find all actual positives) provide a more complete story. 
•	Metric Prioritization: I learned how to choose metrics based on real-world costs. For example, in medical screening, Recall is critical to avoid missing actual cases, whereas in spam filtering, Precision is prioritized to avoid blocking important emails. 
•	F1-Score: I learned that the F1-Score acts as a balance between Precision and Recall, which is useful when you need a single performance metric that accounts for both. 
•	Cross-Validation Benefits: I discovered that cross-validation provides a more robust and trustworthy estimate of true performance by averaging results over multiple "folds" of the data. 
________________________________________
Challenges Faced
•	Data Preprocessing: Handling the initial data cleaning, such as mapping strings to integers and dropping NaN values, was a necessary step before the model could be trained. 
•	Interpreting Errors: Distinguishing between False Positives and False Negatives in the context of the Titanic dataset (predicting survival) required careful thought to ensure the "Died" and "Survived" classes were mapped correctly in the visualization. 
•	Variance in Folds: During cross-validation, I observed that different folds produced different scores (e.g., ranging from 75% to 83%), highlighting why a single train-test split can be unreliable.

