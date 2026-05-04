# Module 3 Lab — Machine Learning Workflow and Types of Learning

This repository contains my completed work for the **Module 3 Lab Exercise: Machine Learning Workflow and Types of Learning**. Where Module 2 was about getting the *tools* set up, Module 3 was about putting those tools together for the first time into the full pipeline that every ML project follows: load a real dataset, split it, train a model on the training half, evaluate it on the held-out half, and interpret what came out the other side.

## Contents

- `Module_03_Lab_Exercise-1.ipynb` — the completed Jupyter notebook with all code cells executed and the reflection written.
- `README.md` — this file.

## What I Did in This Lab

The lab was organized into ten parts. I worked through each one in order inside a single notebook.

In **Part 1**, I read through the definitions of the three main types of machine learning — supervised, unsupervised, and reinforcement — along with their canonical examples (classification and regression for supervised, clustering and dimensionality reduction for unsupervised, and game-playing agents for reinforcement).

In **Part 2**, I imported the libraries the rest of the lab depends on. This was a step up from Module 2: in addition to pandas, NumPy, and matplotlib, I imported seaborn for prettier plots and several pieces of scikit-learn — `load_wine`, `train_test_split`, `LogisticRegression`, `DecisionTreeClassifier`, `accuracy_score`, `classification_report`, `confusion_matrix`, and `StandardScaler`.

In **Part 3**, I loaded the **Wine dataset** with `load_wine()` and converted it into a DataFrame. The dataset has 178 samples, 13 chemical features (alcohol, malic acid, ash, magnesium, total phenols, flavanoids, color intensity, proline, and so on), and three target classes (`class_0`, `class_1`, `class_2`) corresponding to three Italian wine cultivars. I confirmed there were no missing values and that the class balance was reasonable but not perfectly even: 71 samples in class 1, 59 in class 0, and 48 in class 2.

In **Part 4**, I did basic **exploratory data analysis**. I plotted a bar chart of the class distribution side-by-side with a correlation heatmap of the first six features, which exposed a few strong correlations among the chemical measurements that I would not have spotted by reading the numeric values alone.

**Part 5** was the heart of the lab — implementing the full ML workflow as six explicit steps:

- **Step 1 — Data preparation.** I selected four starter features (`alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`) as `X` and `wine_class` as `y`, giving me a feature matrix of shape `(178, 4)`.
- **Step 2 — Data splitting.** I used `train_test_split` with `test_size=0.2`, `random_state=42`, and `stratify=y` to keep class proportions intact. The result was 142 training samples and 36 test samples.
- **Step 3 — Model training.** I trained two models on the same training data: a `LogisticRegression` and a `DecisionTreeClassifier(max_depth=3)`.
- **Step 4 — Model evaluation.** Logistic regression hit **88.9% accuracy** on the test set. The decision tree hit **83.3%**. I printed full per-class precision, recall, and F1 scores using `classification_report`. Logistic regression was perfect on `class_0` (precision and recall both 1.00) but weaker on `class_2` (recall 0.70).
- **Step 5 — Model interpretation.** I plotted a confusion matrix for the winning model so I could see exactly *which* classes were getting confused for which.

In **Part 6**, I worked through a survey of the data types that appear in ML — numerical continuous, numerical discrete, categorical nominal, categorical ordinal, text, and boolean — with a use case for each one.

In **Part 7**, I did a hands-on practice task where I picked a different set of three features (`alcohol`, `color_intensity`, `proline`) and trained a fresh logistic regression model on them. This gave me **83.3% accuracy** — *lower* than the 88.9% I had achieved with the original four features. The lab then prompted me to keep experimenting to see if I could improve performance.

In **Part 8**, I completed an assessment that asked me to label five scenarios as supervised, unsupervised, or reinforcement learning. I got all five correct (predicting house prices → supervised; clustering customers → unsupervised; teaching a robot chess → reinforcement; classifying spam emails → supervised; finding hidden topics in news articles → unsupervised).

In **Parts 9 and 10**, I read through real-world case studies (recommendation systems, fraud detection, medical diagnosis) and a checklist version of the full ML lifecycle from problem definition through deployment and continuous improvement.

Finally, I filled out the reflection cell at the end with my own definitions of the three ML types and my analysis of the model results.

## What I Learned

The biggest takeaway from this lab was that the ML workflow is a **discipline**, not a sequence of clever tricks. There's a fixed shape — load, split, train, evaluate, interpret — and the quality of the work depends much more on whether each step is done correctly than on which fancy algorithm is used at the training step.

A few specific things stuck with me:

- **Why we split the data before training.** Module 2 ended at "load and visualize," so this was the first time I actually had to hold out test data. Doing it via `train_test_split` with `stratify=y` was the cleanest possible introduction: one function call, but built on the entire premise of supervised learning, which is that a model is only as good as its performance on data it has never seen.
- **Accuracy is not the only number that matters.** When `classification_report` printed precision, recall, and F1 per class, I could see something accuracy alone had hidden: the logistic regression model was perfect on `class_0` and good on `class_1`, but its recall on `class_2` was only 0.70. The single accuracy number (88.9%) papered over the fact that 30% of `class_2` wines were being misclassified.
- **More features ≠ better model.** This was the most surprising thing in the lab. My Part 7 model used three features that I'd cherry-picked because they sounded important (alcohol, color intensity, proline) and got *worse* accuracy than the original four-feature model that used some boring chemistry measurements. Feature selection is a real skill, and intuition about which features will be predictive is often wrong.
- **Confusion matrices show you the shape of the error, not just its size.** A single accuracy number doesn't tell you whether the model confuses class 1 with class 2 specifically, or whether it spreads errors evenly. The confusion matrix does, and that pattern is exactly the information you need to decide what to fix next (more data for one class? a different feature? a different algorithm?).
- **Logistic regression is a real model, not a baseline.** I'd absorbed somewhere that logistic regression was the "starter" algorithm and decision trees were "stronger." In this lab logistic regression beat the decision tree by 5.6 percentage points on the same data. The lesson is that there is no universally stronger algorithm — only algorithms that suit some datasets better than others.
- **Stratified splits prevent silent bugs.** With class_2 having only 48 samples, an unstratified random split could easily have left me with very few class_2 examples in the test set, which would have made my test accuracy look much better than it actually was. The `stratify=y` parameter is small but consequential.
- **Reproducibility is a one-line decision.** Setting `random_state=42` everywhere meant my results matched the lab's expected outputs and that I could re-run the notebook from a clean state and get the same numbers. That habit is going straight into every future lab.

## Challenges I Faced

Several places in this lab made me slow down and think.

- **Picking which features to start with.** The Wine dataset has 13 chemical measurements with names like "od280/od315 of diluted wines" and "nonflavanoid phenols." With no chemistry background, I couldn't tell which of those should be most informative. I went with the lab's suggestion of the first four features for the main model; for the Part 7 experiment I picked features whose names I recognized, which turned out to be a worse strategy than picking arbitrarily.
- **Why my Part 7 model was *worse* than the original.** The first time I ran my custom model and saw 83.3% versus the original's 88.9%, I thought I had broken something. Slowing down made me realize the model was fine — those particular three features just carried less signal *together* than the original four did. Untangling "I made a mistake" from "the experiment shows something real" is its own skill.
- **Reading the classification report.** The output is a 5-column table with three rows for the per-class scores plus rows for accuracy, macro average, and weighted average. The first time I looked at it, I had to remind myself what precision and recall actually measure (precision = of the things I called class X, how many really were; recall = of the things that really were class X, how many did I catch). Building that vocabulary is going to take more than one lab.
- **The confusion matrix axes.** It is easy to read a confusion matrix in the wrong direction. I had to consciously remind myself that rows are *true* classes and columns are *predicted* classes, so a number in the off-diagonal `[class_2, class_1]` cell means "a wine that was actually class_2 got predicted as class_1."
- **Class balance versus class imbalance.** Wine has 71 / 59 / 48 samples per class, which is uneven but not extreme. The lab didn't make me do anything special to handle this, but it left me wondering what the threshold is for "imbalanced enough that I need to do something about it" — that's a question I want to come back to.
- **Choosing the right random_state convention.** Multiple cells use `random_state=42` for reproducibility, but I had to be careful that the value was set consistently in the split *and* the models. Forgetting it in one place would have produced subtly different results between runs.

## Results Summary

| Model                | Features                                                  | Test accuracy |
|----------------------|-----------------------------------------------------------|---------------|
| Logistic Regression  | alcohol, malic_acid, ash, alcalinity_of_ash               | **88.9%**     |
| Decision Tree (d=3)  | alcohol, malic_acid, ash, alcalinity_of_ash               | 83.3%         |
| Logistic Regression  | alcohol, color_intensity, proline                         | 83.3%         |

Test set: 36 samples (12 of class_0, 14 of class_1, 10 of class_2).

## Tools and Versions Used

- Python 3
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn (`load_wine`, `train_test_split`, `LogisticRegression`, `DecisionTreeClassifier`, `accuracy_score`, `classification_report`, `confusion_matrix`, `StandardScaler`)
- Jupyter Notebook / Google Colab

## Next Steps

Module 4 is going deeper into exploratory data analysis and data quality. Three things I want to take into it from this lab:

1. **Try every feature, then trim.** Instead of guessing which features will be predictive, I want to start with the full feature set and use a feature-importance method to remove the weak ones. The Part 7 result made it clear that human intuition about feature usefulness is unreliable.
2. **Look at per-class metrics first, accuracy second.** The 88.9% headline number on logistic regression hid a real weakness on `class_2`. Going forward, the classification report is the first thing I want to read.
3. **Try at least one more model type.** Logistic regression beat the decision tree on this dataset, but with only two algorithms compared, I can't tell whether that's because logistic regression suits this data well or because the decision tree was under-tuned. A third algorithm — random forest is the obvious next choice — would let me test which it is.
