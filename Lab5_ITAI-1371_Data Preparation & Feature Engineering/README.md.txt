Module 05 Lab: Data Preparation and Preprocessing
Project Overview
This lab focused on the essential phase of the machine learning workflow: Data Preparation (Preprocessing). Using the classic Titanic dataset, I addressed common real-world data issues such as missing values and non-numeric categorical data to ensure the dataset was ready for a machine learning model.

Key Activities
Initial Data Audit: Loaded the Titanic dataset and identified several columns with missing values (NaN), specifically 'Age' (177 missing), 'Cabin' (687 missing), and 'Embarked' (2 missing).

Handling Missing Values (Imputation):

Targeted the 'Age' column for imputation.

Calculated the median age and used the .fillna() method to replace missing entries.

Categorical Feature Encoding:

Applied One-Hot Encoding using pd.get_dummies() for the 'Sex' and 'Embarked' columns.

Used the drop_first=True parameter to prevent redundant columns (the "dummy variable trap").

Feature Scaling:

Utilized the StandardScaler from scikit-learn to normalize 'Age' and 'Fare'.

Rescaled these features to have a mean of 0 and a standard deviation of 1 to prevent features with larger ranges from disproportionately influencing model training.

What I Learned
The Robustness of Medians: I learned that the median is a more robust imputation strategy than the mean when dealing with skewed data or datasets with outliers, as extreme values can heavily bias the mean.

Converting Text to Math: I solidified my understanding of One-Hot Encoding, which is necessary because machine learning models are mathematical and require numerical inputs rather than raw text strings.

Model-Specific Requirements: I discovered that not all models require the same preparation. For instance, distance-based models (like KNN or SVM) are highly sensitive to feature scales and require scaling, whereas Decision Trees are generally unaffected because they split data based on thresholds.

Challenges Faced
Selective Scaling: One challenge was ensuring that only the appropriate numerical features ('Age' and 'Fare') were scaled, while leaving categorical indicators and the target variable untouched.

Addressing Data Loss: Initially considering whether to drop missing values versus imputing them was a challenge. I learned that dropping rows can lead to significant data loss, making imputation a more effective strategy for maintaining a viable dataset size.

Dummy Variable Trap: Understanding why we drop the first category in One-Hot Encoding (to avoid multicollinearity) required careful attention during the coding phase.