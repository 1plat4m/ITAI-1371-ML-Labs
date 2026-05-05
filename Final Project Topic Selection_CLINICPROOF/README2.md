What your CLINICPROOF project is about

CLINICPROOF is my AI project that simulates a healthcare procedure compliance verification system. It uses machine learning to analyze human activity sensor data, classify what a healthcare worker is doing, and help determine whether a clinical procedure was actually performed correctly. From this project, I learned how to prepare data, train multiple machine learning models, evaluate predictions across different metrics, and connect AI results to real-world patient-safety decisions.

The idea is to help hospitals confirm that healthcare workers are following required safety procedures — hand hygiene, medication administration, surgical safety checklists, fall prevention — before a task is documented as completed.

Instead of relying only on paper logs, manual sign-offs, or honor-system documentation, CLINICPROOF uses wearable sensor data, AI, and per-class compliance metrics to help detect whether a procedure was compliant, requires review, or non-compliant. The project simulates how a real-world system could support hospitals, surgical centers, long-term care facilities, and clinical training programs.

In simple terms:

CLINICPROOF helps verify that clinical procedures were performed correctly, safely, and with sensor-verified evidence — not just a checkbox on a form.

What you learned from the project

Through this project, you learned how machine learning can be used to solve a real patient-safety problem. You worked with the public UCI Human Activity Recognition dataset and used AI concepts to classify human activity into six categories that map to clinical procedural states.

You also learned how to prepare data, train and compare four different models, and evaluate results using visuals like a confusion matrix, per-class precision and recall scores, feature importance charts, and a dangerous-confusion analysis specific to clinical safety.

The biggest lesson is that AI in healthcare is not just about high accuracy. It is about creating a complete, defensible workflow:

data → preprocessing → model → prediction → per-class evaluation → safety verification → decision support

You also learned that the type of error matters as much as the number of errors. In a clinical setting, predicting WALKING when the worker is actually SITTING is a much more dangerous mistake than confusing WALKING_UPSTAIRS with WALKING_DOWNSTAIRS — both still mean the worker is moving. Standard accuracy hides that distinction. CLINICPROOF makes it visible.

How the code works

The code follows these steps:

1. Import libraries

The notebook starts by loading Python tools such as:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

These libraries handle data, machine learning, and visualization.

2. Load the dataset

The project uses the UCI Human Activity Recognition (HAR) dataset, which contains 10,299 sensor recordings from 30 volunteers. Each recording has 561 features extracted from a smartphone's accelerometer and gyroscope at the waist, and is labeled with one of six activity classes:

WALKING
WALKING_UPSTAIRS
WALKING_DOWNSTAIRS
SITTING
STANDING
LAYING

The goal is to use these features to predict the activity, and then map those activities onto clinical procedural states (worker rounding, stationary procedural work, etc.).

3. Prepare the data

The dataset arrives pre-split into training and test sets, but the code applies one important preprocessing step — standardization — so the features have mean 0 and standard deviation 1.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

The scaler is fit only on training data so test-set statistics never leak into the model.

4. Use the pre-supplied subject-aware split

The dataset is already split, but it is split by subject — 21 volunteers in training, 9 in testing. This is important: the model never sees test subjects during training, so the test accuracy reflects how well the model generalizes to people it has never seen.

X_train shape: (7352, 561)
X_test shape:  (2947, 561)

5. Train and compare four models

CLINICPROOF compares four classical machine learning models that span the bias-variance spectrum:

models = {
    'Logistic Regression':   LogisticRegression(max_iter=2000, random_state=42),
    'KNN (k=7)':             KNeighborsClassifier(n_neighbors=7),
    'Linear SVM (C=10)':     LinearSVC(C=10, random_state=42, dual=False),
    'Random Forest (n=200)': RandomForestClassifier(n_estimators=200, random_state=42),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

This teaches each model to recognize the patterns that distinguish the six activities.

6. Make predictions

Each model predicts the activity for every sample in the test set.

y_pred = model.predict(X_test_scaled)

7. Evaluate the models

The code reports overall accuracy, macro-F1, and a full per-class classification report.

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

It also generates a confusion matrix for the winning model:

confusion_matrix(y_test, y_pred)

The Random Forest emerges as the strongest performer, with accuracy in the published UCI HAR benchmark band of 89–96%.

8. Run the dangerous-confusion analysis

This is the step that makes CLINICPROOF different from a generic HAR project. Not every misclassification is equally dangerous in a clinical setting:

Harmless: WALKING_UPSTAIRS confused with WALKING_DOWNSTAIRS (worker is still moving)
Mildly costly: SITTING confused with STANDING (both stationary)
DANGEROUS: a dynamic activity (WALKING) predicted as a static activity (SITTING) — this would mean the system tells the hospital "the worker is rounding" when they are actually stationary

The code calculates this dangerous-confusion rate and checks it against the < 5% safety threshold.

9. Look at feature importance

The Random Forest reports which sensor features mattered most to its predictions. The top features are typically gravity-axis components (orientation: is the body vertical or horizontal?) and frequency-domain energies (rhythm: is the body shaking?), which match what a domain expert would name independently.

How to run the code

To run the CLINICPROOF project:

Open the notebook file (CLINICPROOF_ML_Pipeline.ipynb) in Jupyter Notebook, JupyterLab, Google Colab, or VS Code.
Google Colab is the recommended option because the dataset downloads directly inside the notebook without any setup.
Make sure the required libraries are installed.
Run each cell from top to bottom.
Review the output after each section — especially the model comparison table, the confusion matrix for the winning model, and the dangerous-confusion analysis.
Check the final charts and the success-criteria scorecard at the end.

If using VS Code, open the .ipynb notebook and click Run All.

If a library is missing, install it with:

pip install pandas numpy matplotlib seaborn scikit-learn

If the dataset download fails (due to firewall or offline use), the notebook automatically falls back to a synthetic UCI-HAR-shaped dataset so the entire pipeline still runs end-to-end. In Google Colab this fallback is never used and the real UCI HAR data loads directly.
