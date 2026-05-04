# Module 9 Lab — Ensemble Methods

This repository contains my completed work for the **Module 9 Lab: Ensemble Methods**. Module 8 ended on the bias-variance tradeoff and the realization that overfit models — high variance — could be calmed down with the right techniques. Module 9 introduces one of the most important of those techniques: **ensembles**, where a "committee" of imperfect models, taken together, beats any single model on its own.

## Contents

- `Module_09_Lab_-_Ensemble_Methods.ipynb` — the completed Jupyter notebook with all four code cells executed and the Knowledge Check answered.
- `README.md` — this file.

## What I Did in This Lab

The lab returned to the **Iris dataset** I had used in Module 2 — 150 samples, four features, three target species — but this time the goal was not to load and visualize it. The goal was to compare a single decision tree against a random forest on the same data and see whether the ensemble actually delivered the gains the theory promised.

The lab was organized into five parts plus a Knowledge Check.

In **Part 1 — The Wisdom of the Crowd**, I read through the conceptual setup. The core idea behind ensembles is that a large, diverse group of models, asked the same question, will collectively produce a more accurate answer than any single model — the same way a crowd's average guess at "how many jellybeans are in this jar?" is often closer to the truth than any individual guess. The lab introduced two ensemble strategies: **bagging** (training many models on different random subsets of the data and averaging their predictions, which is what a Random Forest does) and **boosting** (training models sequentially, where each new model focuses on the errors of the previous ones, as in Gradient Boosting and AdaBoost).

In **Part 2 — Setup**, I imported pandas, the Iris loader, `train_test_split`, `DecisionTreeClassifier`, `RandomForestClassifier`, and `accuracy_score`. I loaded Iris with `load_iris()` and split it 70/30 with `random_state=42`, leaving 105 training samples and 45 test samples.

In **Part 3 — A Single Decision Tree**, I trained a `DecisionTreeClassifier(random_state=42)` and evaluated it on the test set. The result was **100.00% accuracy**. The lab had warned me that single trees are prone to overfitting, so I went into Part 4 expecting the Random Forest to outperform it — and was surprised by what came next.

In **Part 4 — The Random Forest Ensemble**, I trained a `RandomForestClassifier(n_estimators=100, random_state=42)` — a committee of 100 trees, each trained on a random subset of the data and a random subset of the features. The Random Forest also got **100.00% accuracy**. Both models, despite being conceptually very different, produced an identical perfect score on this test set. This was the most interesting honest finding of the lab, and I addressed it head-on in the Knowledge Check.

In **Part 5 — Feature Importance**, I extracted the `feature_importances_` attribute from the trained Random Forest and plotted the four features ranked by their importance. The numbers came out cleanly:

- **petal width (cm): 0.434**
- **petal length (cm): 0.417**
- **sepal length (cm): 0.104**
- **sepal width (cm): 0.045**

Petal measurements together accounted for about 85% of the model's discriminative power, while sepal measurements together carried only about 15%. This is the kind of result that takes a model from a black box to a story you can tell: the model has effectively rediscovered what botanists have known for over a century — iris species are most reliably told apart by their petals.

In the **Knowledge Check**, I answered three questions: the main idea behind ensemble methods (errors cancel out across diverse models, so the committee is more stable and accurate than any individual); which model performed better and whether I expected it (both tied at 100%, which can happen on a small, well-separated dataset like Iris, but I still expected the Random Forest to be at least as reliable as a single tree because of its lower variance); and which features were most important (petal width and petal length, which matches what I'd expect from real-world knowledge of flowers).

## What I Learned

The biggest takeaway was that **ensembles are not a magic accuracy boost — they are a variance reduction technique**. On easy, small, well-separated datasets like Iris, a single tree can already nail the problem and there's no headroom left for an ensemble to improve. On harder datasets — noisy, high-dimensional, ambiguous — the same single tree would be brittle, and the ensemble's averaging mechanism would make a much bigger difference. Knowing *when* an ensemble will help is as important as knowing how to build one.

A few specific things stuck with me:

- **The 100%-vs-100% result is itself a lesson.** My instinct after reading the lab text was that the Random Forest "should" win, and when it didn't, my first thought was "did I do something wrong?" Slowing down made it clear: there are 45 samples in my test set, the three Iris species are nearly linearly separable in petal-space, and a single decision tree with no pruning can easily build a rule that captures all 45 correctly. There was no room for the ensemble to add value because the single tree wasn't actually struggling. The right takeaway isn't "ensembles don't help" — it's "ensembles help most where single models are weakest."
- **Bagging and boosting are different cures for different diseases.** Bagging (Random Forest) is variance reduction: build many high-variance models on different data subsets and average them out, and the noise cancels. Boosting (Gradient Boosting, AdaBoost) is bias reduction: train models sequentially, each one fixing the previous one's mistakes, and the model gets progressively more capable. They're not interchangeable — they target opposite ends of the bias-variance tradeoff I learned in Module 8.
- **Feature importance is a side benefit that pays for itself.** Random Forest gives you `feature_importances_` essentially for free as a by-product of how the trees are built. That's something a logistic regression doesn't give you in quite the same form, and it's reason enough to try a Random Forest as a diagnostic tool even if you ultimately ship a different model. Knowing that petal measurements explain 85% of the discrimination on Iris is information I can use — it tells me that if I were collecting Iris data in a hurry, sepal measurements would be the ones I could get away with skipping.
- **Two random subsets at once.** A common misunderstanding I had to clear up: a Random Forest doesn't just train each tree on a random subset of the *data*. It also trains each tree on a random subset of the *features*. Both kinds of randomness are essential. Without random feature subsets, every tree in the forest would tend to lean on the same dominant feature and the ensemble would just be many copies of essentially the same tree. The two layers of randomness are what produces the diversity that makes the wisdom-of-the-crowd math work.
- **`n_estimators=100` is a hyperparameter with a tradeoff.** More trees usually means a more stable forest, up to a point. But more trees also means more compute time at training and prediction. On Iris, 100 trees finishes in milliseconds. On a real dataset with 100,000 rows and 50 features, the same parameter is a meaningful cost decision.
- **Connecting back to Module 8.** Random Forest is exactly the kind of tool the Module 8 lab was hinting at without naming. A single deep decision tree is a high-variance model. Averaging 100 of them — each fit to slightly different data — is the bagging mechanism that drops variance without raising bias much. So Module 9 is not really a new topic; it's a concrete tool for the problem Module 8 framed.

## Challenges I Faced

- **Interpreting the 100%-vs-100% result honestly.** It would have been easy to write up the lab as if the Random Forest "won" — or, alternatively, to declare the experiment a failure because both models scored the same. Neither framing is honest. The truthful version is that the Iris dataset is too easy for the difference between a single tree and a 100-tree forest to show up on the test set, and admitting that openly is more useful than hiding it.
- **Knowing what to do when results don't match expectations.** When my Random Forest accuracy didn't beat the single tree, my first instinct was to re-check my code. After confirming there was no bug, the next instinct was to question the experimental setup. The right move is the second one — but it took deliberate effort to not just keep tweaking parameters until I got the "expected" outcome.
- **Reading `feature_importances_` correctly.** The four numbers (0.434, 0.417, 0.104, 0.045) sum to 1.0 by construction, which makes them comparable proportions. But comparing relative importance is one thing; deciding what to *do* with it is another. The two petal features are nearly tied — does that mean they're redundant (one would be enough), or genuinely independent signals (both worth keeping)? Feature importance answers "which features matter," but not "which features could be dropped." That's a distinction I want to come back to.
- **The horizontal bar plot orientation.** The plot uses `plt.barh` with features sorted by `np.argsort(importances)`, which puts the *most* important feature at the *top* of the chart. The first time I read it I had to slow down to confirm the direction — visualization conventions for "ranked importance" are not universal, and a quick squint can have you read the chart upside down.
- **The `random_state=42` in two places.** Both the train/test split and the Random Forest constructor took `random_state=42`. Either one being different would have produced slightly different numbers, and a careful reader of the notebook would want to know that the same seed is used in both spots. Tracking which random source affects which result is something I'm still building intuition for.
- **Resisting the urge to call this lab "trivial."** It's a 4-cell notebook with a foregone conclusion (ensemble theory is well-known), and the test result was a tie. But the most important things in the lab — the conceptual difference between bagging and boosting, the feature-importance attribute, the realization that ensembles target variance specifically — are conceptual rather than experimental. Treating them as the real content rather than the runtime output is what actually matters.

## Results Summary

| Model                     | Test accuracy | Notes                                                                  |
|---------------------------|---------------|------------------------------------------------------------------------|
| Decision Tree (single)    | 100.00%       | Default settings, `random_state=42`                                    |
| Random Forest (100 trees) | 100.00%       | `n_estimators=100`, default depth, `random_state=42`                   |

**Random Forest feature importance** (sums to 1.0):

| Feature              | Importance | Share |
|----------------------|------------|-------|
| petal width (cm)     | 0.4340     | 43%   |
| petal length (cm)    | 0.4173     | 42%   |
| sepal length (cm)    | 0.1041     | 10%   |
| sepal width (cm)     | 0.0446     | 4%    |

Test set: 45 samples (15 per Iris species, by stratified split convention).

## Tools and Versions Used

- Python 3
- pandas, NumPy, matplotlib
- scikit-learn (`load_iris`, `train_test_split`, `DecisionTreeClassifier`, `RandomForestClassifier`, `accuracy_score`)
- Jupyter Notebook
- Iris dataset, loaded via `sklearn.datasets.load_iris()`

## Next Steps

Module 10 should move into model evaluation and hyperparameter tuning — the techniques that actually let you tell two seemingly-tied models apart, and that get pulled out of the toolbox the moment you move to a harder dataset. Three things I want to take into it:

1. **Re-run the comparison on a harder dataset.** The Iris result was a tie because the problem is too easy. The same comparison on the Titanic data (with `Sex`, `Pclass`, `Age`, `Fare` as features after Module 5's preparation) will almost certainly produce a meaningful gap between a single tree and a Random Forest. That experiment will show me what an ensemble actually buys.
2. **Bring back the learning curves from Module 8.** Plotting a learning curve for the single tree and the Random Forest on the same dataset would let me visually confirm what the theory says: the forest has a smaller train-validation gap (less variance) than a single deep tree.
3. **Try a boosting model side-by-side.** Random Forest is bagging; Gradient Boosting is the other major ensemble strategy. Running a `GradientBoostingClassifier` on the same data and comparing its results to the Random Forest will give me concrete intuition for the bagging-vs-boosting tradeoff rather than only the textbook version.


