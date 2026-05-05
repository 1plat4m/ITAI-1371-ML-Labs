# FIELDPROOF™ — Sensor-Verified Human Task Execution for Oil & Gas Operations

This repository contains my **Midterm Project** for ITAI 1371, Spring 2026: a complete machine learning pipeline that demonstrates how human activity recognition can be applied to verifying that field workers in oil & gas operations actually perform safety-critical tasks correctly. The earlier modules in my portfolio (Modules 2 through 5) built up the supervised ML toolkit one piece at a time — load data, prepare it, run the workflow — and FIELDPROOF is where those pieces had to come together on a real domain problem with real stakes for the first time.

## Contents

- `FIELDPROOF_ML_Lab.ipynb` — the Jupyter notebook with the full ML pipeline (synthetic dataset generation, EDA, three trained models, evaluation, feature importance, end-to-end use cases).
- `FIELDPROOF_Abstract.docx` — the original project abstract that scoped the work.
- `README.md` — this file.

## What I Did in This Project

The full FIELDPROOF vision is a multi-system platform — wearables on field workers, equipment-mounted sensors, ML inference on the rig, and an immutable compliance ledger that produces sensor-verified evidence of every safety-critical procedure executed in the field (lockout/tagout, confined-space entry, PPE checks, maintenance verification, training certification, ESG audits). That whole stack belongs to a real product roadmap, not a one-semester course. The midterm project deliberately scopes down to the **technical-feasibility core**: can a classical ML pipeline, trained on sensor-derived features that mirror what a real wearable would emit, distinguish between compliant, non-compliant, and review-needed task executions at an accuracy that would make automated compliance verification possible?

If yes, the bridge to a real field deployment becomes a question of data collection (consented IMU traces from real workers on real rigs) and protocol mapping (each task type — LOTO, valve adjust, confined-space entry — gets its own labeled dataset), not algorithmic invention.

The pipeline walks through the same workflow I'd practiced across Modules 3 through 5: data loading and quality checks, exploratory analysis, train/test split, three trained models compared head to head, full evaluation including per-class metrics and feature importance. Because no public dataset of *oil & gas worker task executions with verified compliance labels* exists, the project uses a **synthetic dataset of 240 task executions** with eight sensor-derived features that match what the FIELDPROOF abstract specifies as the on-device codification output:

- `task_duration_sec` — time-on-task captured by the wearable
- `motion_intensity` — IMU-derived movement magnitude (acceleration RMS)
- `posture_score` — proportion of time in correct posture
- `step_sequence_match` — fraction of required steps executed in correct order
- `proximity_correct_zone_pct` — percent of task time spent in the correct geofenced zone
- `ppe_compliance_score` — detected PPE coverage
- `tool_engagement_sec` — verified time the correct tool was actively used
- `idle_time_sec` — sensor-flagged idle / no-motion seconds

Each task execution is labeled as one of three compliance outcomes: **Compliant** (all signals strong), **Requires_Review** (borderline — most signals OK but one or two soft signals off), or **Non_Compliant** (shortcuts, wrong sequence, out-of-zone, missing PPE). The class definitions are deliberately patterned after how a real field-safety supervisor would categorize the same outcomes in a written audit, which makes the model's eventual predictions human-readable in a way that matters for ESG reporting and legal defense.

The three models compared were Logistic Regression, Decision Tree, and Random Forest — chosen to span the bias-variance spectrum in the same pattern Module 8 would later teach me to think about explicitly. The Random Forest emerged as the strongest performer, and crucially, its feature importance ranking mapped onto signals a domain expert would have nominated by hand: posture, sequence-match, and PPE compliance dominated, while idle time and raw motion intensity carried less discriminative weight than I would have guessed before training the model.

## Results

The three-class classification problem turned out to be cleanly separable on the synthetic data — the class definitions were designed to be physically meaningful, and the models recovered the structure as expected. The performance ordering matched what the bias-variance theory would predict.

| Model | Test accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Logistic Regression | ~88% | ~0.87 | Baseline — confirms the problem is mostly linearly separable |
| Decision Tree (depth 4) | ~83% | ~0.82 | Useful contrast — single tree overfits the boundary regions |
| **Random Forest (n = 100)** | **~96%** | **~0.95** | Final selected model |

(Exact numbers depend on the random seed used for synthetic data generation. The notebook is reproducible with `np.random.seed(42)`.)

The Random Forest's confusion matrix surfaced something worth flagging in the report: the most common error wasn't between Compliant and Non-Compliant — those two extremes are easy to tell apart — but between **Requires_Review** and the two adjacent classes. The model correctly recognized that "borderline" task executions sit on the boundary between the strict-pass and strict-fail categories, and the confusions reflected that geometry rather than random noise. In the field-deployment translation, this is exactly the right behavior: a Requires_Review prediction is supposed to route the task to a human supervisor for adjudication, so having the model push borderline cases into that bucket — rather than confidently mislabeling them as Compliant — is the safer failure mode.

The Random Forest's feature importance ranking told a clean story too. The top three features were `step_sequence_match`, `posture_score`, and `ppe_compliance_score` — collectively accounting for roughly 65% of the model's discriminative power. Those three features map onto exactly the kinds of cues a field-safety supervisor would scan for in a manual audit: did the worker do the steps in the right order, did they hold the right body position throughout, and did they wear the correct PPE? When a Random Forest's most important features match what a domain expert would have nominated by hand, the model has earned a degree of trust that purely accuracy-based evaluation cannot give it.

## What I Learned

The biggest takeaway from FIELDPROOF was that **a synthetic dataset is a legitimate research tool when you know what you're doing with it.** Because the dataset was synthetic, every claim about FIELDPROOF's clinical-quality readiness has to be conditional on a future data-collection effort with consented real workers — and naming that gap honestly in the writeup turned out to be more valuable than pretending I had access to data I didn't. The synthetic data wasn't a shortcut; it was the right tool for proving that *the modeling approach* is sound while being upfront that the *deployment readiness* is not yet demonstrated. That distinction — between "the algorithm works" and "the system is ready to ship" — is one I'll carry into every future project.

A few more specific things stuck with me:

- **Class definitions matter as much as features.** The three-class structure I built (Compliant, Requires_Review, Non_Compliant) wasn't arbitrary — it was designed to mirror how a field-safety supervisor categorizes outcomes in a written audit. That meant the model's predictions translated directly into a human-readable compliance verdict without any additional post-processing. Designing the label space *backward* from the deployment context, rather than forward from the data, is a habit I want to keep.
- **The bias-variance arc is visible even before you know what to call it.** I built FIELDPROOF before reaching Module 8, but the three-model comparison I ran (Logistic Regression → Decision Tree → Random Forest) is exactly the same arc Module 8 later taught me to think about explicitly. Logistic Regression set the baseline. The single Decision Tree showed what a low-bias / high-variance model looks like (it overfit the borderline cases). Random Forest reduced the variance via bagging and won. By the time I sat down for the Module 8 lab, I'd already done the experiment.
- **Random Forest's feature importance is the "earned trust" mechanism.** In a domain like industrial safety, an audit committee or insurance underwriter is going to ask *why* a model flagged a task as non-compliant. A model whose top features (sequence match, posture, PPE) map onto signals a human supervisor would name independently is much easier to defend than a model that depends on weird high-dimensional interactions. This is the same insight I would later carry into the CLINICPROOF final project: classical ensembles win on interpretability, and that often matters more than the last 1–2 percentage points of accuracy a black-box network would buy.
- **Three classes is harder than two — but the right call.** A binary Compliant/Non-Compliant split would have been mathematically easier and would have produced higher headline accuracy. But it would have erased the most operationally useful state: Requires_Review. The whole point of an automated compliance system is that it lets humans focus on the cases that need human judgment, and that requires the model to *say* it doesn't know. Adding the third class made the modeling slightly harder and the deployment story dramatically more honest.
- **Connecting the project back to the course modules made the whole thing coherent.** Module 5's data preparation showed up in the careful synthetic-data generation, the train/test split, and the categorical encoding. Module 4's EDA showed up in the per-class feature distribution plots, which were the visual case for ML before any model was trained. Modules 8 and 9 (which I hadn't reached yet at midterm time) would later give me the vocabulary for what I'd already done — variance reduction via bagging, the bias-variance tradeoff. The midterm wasn't separate from the modules; it was where the early modules got cashed in.
- **Real-world ML is mostly NOT the modeling step.** I spent more time designing the synthetic-data generator (deciding what the three classes should mean physically, what feature distributions they should produce, how much overlap was realistic) than I did fitting the three models combined. The `model.fit()` calls took milliseconds. Designing the problem so it actually mapped onto the FIELDPROOF use case took most of the project.

## Challenges I Faced

- **Naming the gap between the vision and the deliverable honestly.** The FIELDPROOF abstract makes claims about LOTO verification, confined-space entry, maintenance proof-of-work, training certification, and ESG audits. None of those are something I actually built. The temptation was to pretend the synthetic-data pipeline was somehow a partial implementation of that whole vision. The truthful version is that I built a feasibility-check ML pipeline whose scope is "can sensor-derived features distinguish three compliance outcomes?" and that the answer being yes is a starting condition for the larger system, not a fragment of it. Saying that openly was harder than dressing it up.
- **The dataset being entirely synthetic.** Every claim about field-deployment readiness in this project is conditional on a future data-collection effort with consented real workers, because there's no public dataset of oil & gas worker task executions with verified compliance labels. The single largest source of uncertainty is the gap between my synthetic data and the noisy, worker-to-worker-variable, PPE-occluded reality of a real rig. Naming that as the most important next step (rather than burying it in a footnote) was the right call but it felt like undermining my own headline number.
- **Designing the three-class structure to be physically meaningful.** The three classes — Compliant, Requires_Review, Non_Compliant — had to be more than arbitrary cluster centers in feature space. Each class needed to correspond to a realistic worker behavior pattern that a domain expert would recognize. Designing the synthetic generator so that "compliant" tasks looked like high posture scores + correct sequence + high PPE + low idle, and "non-compliant" tasks looked like the systematic opposite (shortcuts, wrong sequence, out-of-zone), and "requires review" looked like the genuinely ambiguous middle, took several iterations. Class definitions are easy to write down and hard to make believable.
- **The Decision Tree's surprising weakness.** Going in, I expected the Decision Tree to be a respectable middle-ground performer. It came in last (~83%, behind Logistic Regression at ~88%). The reason is that with only 240 samples and three classes, a single tree has too much freedom to chase noise in the borderline Requires_Review cases. A Random Forest fixes that by averaging across 100 trees on different bootstrap samples — but understanding *why* the single tree was the weakest, not just *that* it was, took deliberate thinking that the Module 9 lab would later confirm with theory.
- **Choosing how much overlap to put between classes.** If the three classes are too clearly separated in feature space, the models all hit 100% accuracy and the comparison becomes meaningless. If they overlap too much, even the Random Forest can't recover the structure. Finding the right amount of overlap — enough that the bias-variance differences between models showed up, not so much that the problem became unlearnable — was a tuning step I didn't expect to need. It was a real lesson in synthetic data design: the experiment is meaningful only if the problem sits in the difficulty band where models actually have to work.
- **The temptation to skip the Requires_Review class entirely.** Two-class problems are simpler and produce cleaner metrics. The three-class structure was operationally correct (a real compliance system needs a "send to human review" output) but it cost me complexity in evaluation and a couple of points of headline accuracy. Resisting the simplification was the right call but it required actively trading expected metric performance for operational fidelity.

## Tools and Versions Used

- Python 3 in Google Colab
- pandas, NumPy, matplotlib, seaborn
- scikit-learn (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `train_test_split`, `accuracy_score`, `classification_report`, `confusion_matrix`)
- Synthetic dataset of 240 task executions across three compliance classes, eight sensor-derived features, generated with `numpy.random` and `np.random.seed(42)` for reproducibility

## What I Would Improve in Future Work

The five things on my list, in priority order:

1. **Collect real field data, even at small scale.** A pilot with even ten workers performing a single task type — say, lockout/tagout — for a week would convert the feasibility argument into a deployment argument. This is the single largest source of uncertainty and the one that pays off most.
2. **Move from windowed features to sequential models.** The synthetic features collapse a multi-step procedure into a static feature vector, but real LOTO has a sequence (apply lock → verify zero energy → tag → log). A 1D CNN or LSTM on raw IMU signals could learn that ordering directly, where the engineered-feature approach cannot.
3. **Add the multi-sensor fusion the abstract describes.** The win for industrial compliance isn't better IMU classification alone — it's fusing IMU with proximity beacons (was the worker in the right zone?), tool telemetry (did the right tool get powered on?), and equipment state (did the lock actually engage?). The full design uses all of these together; the midterm uses only the IMU/posture/PPE features.
4. **Run leave-one-worker-out cross-validation once real data exists.** The honest measure of generalization to new workers is per-worker performance variance, and that test is meaningless on synthetic data. The moment real data comes in, that's the first evaluation to run.
5. **Build the supervisor-facing dashboard.** A model that emits a verdict is half a product. The other half is the interface a field-safety supervisor uses to review the Requires_Review cases, override predictions when needed, and feed those overrides back into retraining. That dashboard is what makes FIELDPROOF a deployable system rather than an isolated classifier.

## Acknowledgments

This project sits on top of several bodies of work that deserve direct credit. The **UCI Human Activity Recognition** literature (Anguita et al., 2013) established the engineered-feature approach to IMU classification that I adapted for the FIELDPROOF feature set. **OSHA** publishes the regulatory framework (29 CFR 1910.147 for lockout/tagout, 1910.146 for confined-space entry) that defines what "compliant" actually means for the kind of tasks FIELDPROOF would verify in production. The synthetic-data approach itself was inspired by `sklearn.datasets.make_classification` and `make_blobs`, which are the same family of tools used in Module 10's clustering work — I used the same logic to generate domain-realistic data when no real-world public dataset existed.

---
*Companion to BITWEAR (an earlier sensor-verification exploration in cryptocurrency mining) and CLINICPROOF (the final project, applying the same general approach to healthcare). Together the three projects show the same technical lineage — body-worn sensors plus supervised ML for human-activity verification — applied across three very different settings: oil & gas, cryptocurrency, and healthcare.*


