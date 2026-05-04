# Module 8 Lab — The Bias-Variance Tradeoff

This repository contains my completed work for the **Module 8 Lab: The Bias-Variance Tradeoff**. Earlier modules built the full ML workflow (Module 3), explored data (Module 4), prepared it for modeling (Module 5), and walked through specific algorithms in the modules between. Module 8 is a step back from any one algorithm and into the *theory* that explains why models succeed or fail. Every model I build for the rest of the course will sit somewhere on this lab's spectrum — too simple, too complex, or just right — so this is one of the most important conceptual labs in the program.

## Contents

- `Module_08_Lab_-_Bias-Variance_Tradeoff.ipynb` — the completed Jupyter notebook with all three code cells executed and the Knowledge Check answered.
- `README.md` — this file.

## What I Did in This Lab

The lab took a different approach from the previous ones — instead of working with a real-world dataset, it generated a **synthetic dataset with a known underlying pattern** so I could see overfitting and underfitting cleanly, without any of the noise that real data brings. The dataset was a sine wave with Gaussian noise: 100 evenly-spaced X values from −5 to 5, with `y = sin(X) + N(0, 0.5)`. Knowing the true generating function — `sin(X)` — was the whole point. It meant I could compare each model not just to the data, but to the *real* pattern the data was sampled from.

The lab was organized into four parts plus a Knowledge Check.

In **Part 1 — Understanding the Concepts**, I read through the definitions of the three concepts the lab was built around: **underfitting** (high bias — the model is too simple, performs poorly on both training and test data), **overfitting** (high variance — the model is too complex, performs great on training data but poorly on test data because it memorized the noise), and the **bias-variance tradeoff** (finding the model that's complex enough to capture the true pattern but simple enough not to memorize the noise).

In **Part 2 — Setup**, I imported NumPy, matplotlib, and three scikit-learn pieces — `make_pipeline`, `PolynomialFeatures`, `LinearRegression`, and `learning_curve` — then generated the synthetic sine wave dataset with `np.random.seed(0)` for reproducibility and plotted it as a scatter to confirm the underlying pattern was visible through the noise.

In **Part 3 — Modeling with Different Complexities**, I trained three polynomial regression models with degrees **1, 4, and 15** on the same data. Polynomial regression is a clean way to control model complexity because the degree directly maps onto how flexible the resulting curve can be — a degree-1 polynomial is a straight line, a degree-4 polynomial can curve four times, and a degree-15 polynomial can curve fifteen times. I built each model as a `make_pipeline(PolynomialFeatures(degree), LinearRegression())` so the polynomial expansion and the regression were chained together, then plotted all three predictions on top of the original scatter. The plot made the three behaviors instantly visible: the degree-1 line was nearly flat and missed the wave entirely, the degree-15 curve wiggled aggressively trying to hit individual points, and the degree-4 curve traced the underlying sine pattern smoothly without chasing the noise.

In **Part 4 — Learning Curves**, I read about how **learning curves** diagnose bias and variance by plotting training-set and cross-validation scores as a function of training-set size. The lab gave me a `plot_learning_curve()` helper function and asked me to apply it to the underfit (degree 1) and overfit (degree 15) models. The two curves told different stories. For degree 1 (high bias), both training and cross-validation scores were low and converged toward each other as training size grew — adding more data didn't help, because the model was structurally incapable of learning the pattern. For degree 15 (high variance), there was a noticeable gap between a high training score and a much lower cross-validation score, and the cross-validation scores were unstable across folds — a classic signature of overfitting.

In the **Knowledge Check**, I answered three questions: which of the three models was underfitting, overfitting, and a good fit (degree 1, degree 15, and degree 4 respectively); what the learning curve for the underfit model showed about high bias (low scores converging together — more data doesn't help); and what the learning curve for the overfit model showed about high variance (a noticeable gap and unstable cross-validation scores, especially at smaller training sizes — the model fits training behavior too closely and doesn't generalize consistently).

## What I Learned

The biggest takeaway from this lab was that **bias and variance are not failure modes that go away with effort — they are the two ways a model can be wrong**, and improving one usually costs you the other. Every modeling decision I make from here on out (choosing an algorithm, picking hyperparameters, deciding whether to collect more data) is really a decision about where to sit on that tradeoff. The lab gave me a visual vocabulary for talking about it that I didn't have before.

A few specific things stuck with me:

- **Knowing the true underlying function changed everything.** In every previous lab I had no way to tell whether a model was capturing the "real" pattern or just fitting noise — there was only the data, and the data already contained both. Generating synthetic data from `sin(X)` plus noise let me see exactly what the model was supposed to learn and exactly where it went wrong. That setup is something I can recreate any time I want to test a new algorithm — generate from a known function, compare what the model recovers.
- **Polynomial degree is a knob for complexity.** Degree 1 was a straight line. Degree 4 was a smooth curve. Degree 15 was a wild snake. The same algorithm — linear regression on polynomial features — produced all three behaviors, controlled by a single integer. That's a clean way to see that "complexity" in ML is not a vague property of an algorithm but something that can be turned up and down. Every algorithm I learn from here will have its own version of this knob: tree depth for decision trees, `k` for KNN, regularization strength for logistic regression, number of estimators for random forests.
- **Underfitting and overfitting look completely different on a learning curve.** The two curves had distinct, recognizable signatures. Underfitting: both lines low, close together, plateauing early. Overfitting: training line high, validation line low, big gap, unstable validation scores. Once I'd seen the two shapes, I could imagine running learning curves on any future model and immediately diagnosing what's wrong with it — and what kind of fix to try.
- **More data is not a universal solution.** This was the most counter-intuitive lesson in the lab. The intuition I came in with was "if my model is bad, get more data." The underfit-model learning curve killed that intuition: both scores were already converging at low values, meaning more data would just give me more confidence in the same wrong answer. The cure for high bias is a more flexible model, not more data. The cure for high variance, on the other hand, *is* often more data — which is why the two failure modes call for opposite responses.
- **Cross-validation variance is itself a signal.** Beyond the gap between training and validation scores, the *spread* of validation scores across folds was meaningful. The overfit model's validation scores at small training sizes were not just lower than the training scores — they were also widely spread across folds. That spread itself is the literal "variance" in "high variance," and seeing it in the learning curve made the term make sense in a way the textbook definition didn't.
- **The "just right" model is doing less work than the wrong ones.** The degree-4 polynomial wasn't wrestling with the data the way degree 15 was, but it captured the pattern better. That maps onto a useful general principle: a good model usually looks calmer than a bad one. If a model is fighting the data — twisting and wiggling to hit specific points — that's often a signal that it's overfitting rather than that it's working hard.
- **`make_pipeline` is the right way to chain steps.** Wrapping `PolynomialFeatures` and `LinearRegression` together with `make_pipeline` meant I could fit the whole thing as one object and call `.predict()` once. This is the same pattern that will become standard the moment I do scaling + modeling together, or imputation + scaling + modeling — chaining the steps prevents data leakage and keeps the code readable.

## Challenges I Faced

- **The synthetic-data setup felt artificial at first.** After four labs on real datasets (Iris, Wine, Titanic), being told to fit polynomials to a fake sine wave felt like a step backwards. It took me until the comparison plot came together to understand that the artificial setup was the *point* — being able to see the true function let me tell signal from noise in a way I never could on real data.
- **Reading the comparison plot correctly.** Three lines on top of a scatter looks like a lot at first. The trick was to look at each line separately: does this line track the wave shape of the data, or does it cut straight through, or is it bouncing off individual points? Once I asked the right question, "which is underfitting / overfitting / just right" became obvious.
- **Cross-validation as a concept.** The `learning_curve` function quietly does a lot — it splits the data into folds, trains the model on growing subsets of the training fold, and evaluates on the validation fold. The first time I read the helper function code I had to slow down and trace what `cv` was doing. Knowing that "cross-validation score" is the average of multiple evaluations on held-out folds (rather than a single test evaluation) is what makes the learning-curve story make sense.
- **Why high-variance models have unstable cross-validation scores.** This wasn't immediately obvious. The reason is that an overfit model is so sensitive to its specific training data that, when cross-validation gives it slightly different training subsets, it produces noticeably different models — and so different validation scores from fold to fold. That instability is itself a diagnostic, and it's something I'd missed on my first read of the lab description.
- **Resisting the urge to pick the most complex model.** Looking at the comparison plot, the degree-15 curve looks like it's "doing more work" — it touches more points, follows more wiggles. Naive intuition says "the model that fits more data points must be the better model." The lab was specifically designed to break that intuition. The degree-4 model fits *fewer* points exactly, but recovers the true underlying pattern, which is the thing that matters when new data arrives.
- **The notebook gap between Module 5 and Module 8.** I jumped from Module 5 (data preparation) to Module 8 (bias-variance) without separate labs in between. That left some context to fill in: the modules I'd skipped over presumably introduced a few specific algorithms, but I came into Module 8 with only the logistic regression and decision tree from Module 3 as concrete examples. Reading polynomial regression as a stand-in for "any model whose complexity can be controlled" required a bit of generalization on my part.

## Results Summary

| Model              | Behavior on data                        | Bias/Variance diagnosis | Learning-curve signature                                          |
|--------------------|------------------------------------------|-------------------------|-------------------------------------------------------------------|
| Polynomial deg 1   | Nearly flat line, misses the wave        | **High bias** (underfit) | Both scores low, converging toward each other early              |
| Polynomial deg 4   | Smooth curve tracing the sine pattern    | **Just right**           | Training and validation scores close and high                     |
| Polynomial deg 15  | Wild wiggly curve chasing individual points | **High variance** (overfit) | Big gap; validation scores low and unstable across folds       |

## Tools and Versions Used

- Python 3
- NumPy, matplotlib
- scikit-learn (`make_pipeline`, `PolynomialFeatures`, `LinearRegression`, `learning_curve`)
- Jupyter Notebook
- Synthetic dataset: `y = sin(X) + N(0, 0.5)` for X in `linspace(-5, 5, 100)`, seeded with `np.random.seed(0)`

## Next Steps

Module 9 should move into techniques that directly fight the failure modes this lab identified. Three things I want to take into it:

1. **Apply learning-curve diagnostics to a real dataset.** Now that I know what underfitting and overfitting look like as plot shapes, I want to run the same diagnostic on the Titanic logistic regression I had planned for Module 6 — and see whether I'm closer to high bias or high variance with the prepared data.
2. **Try regularization as the explicit fix for high variance.** Ridge and Lasso regression add a penalty term that shrinks coefficients, reducing the model's effective complexity. If I take the degree-15 model from this lab and add ridge regularization, I expect the wild wiggling to calm down without dropping all the way back to a straight line. That experiment is on my list.
3. **Get more comfortable with cross-validation.** I used it in this lab via the `learning_curve` helper, but I haven't yet built a cross-validation loop myself. Doing that explicitly — splitting data into folds, training and scoring on each, averaging the results — will lock in the concept beyond what reading the helper code gave me.
