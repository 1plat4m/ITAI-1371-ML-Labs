# CLINICPROOF™ — Sensor-Verified Medical Procedure Compliance

This repository contains my **Final Project** for ITAI 1371, Spring 2026: a complete machine learning pipeline that demonstrates how human activity recognition can be applied to verifying clinical procedure compliance in healthcare settings. The earlier modules in my portfolio (Modules 2 through 10) built up the toolkit one piece at a time — load data, prepare it, run the workflow, understand bias-variance, ensemble, cluster — and CLINICPROOF is where those pieces had to come together on a real problem with real stakes.

## Contents

- `CLINICPROOF_ML_Pipeline.ipynb` — the Jupyter notebook with the full ML pipeline (data loading, preprocessing, EDA, four trained models, evaluation, feature importance, confusion analysis).
- `CLINICPROOF_Final_Project_Report.pdf` — the seven-section final report (Introduction, Problem Definition, Approach, Results, Discussion, Conclusion, Supplementary Information).
- `CLINICPROOF_Abstract.docx` — the original project abstract that scoped the work.
- `CLINICPROOF_PRD.md` — the Product Requirements Document specifying dataset, models, evaluation criteria, and timeline.
- `README.md` — this file.

## What I Did in This Project

The full CLINICPROOF vision is a multi-system platform — wearables on healthcare workers, environmental sensors (soap dispensers, NFC patient wristbands, RFID medication carts), real-time ML inference, and a blockchain compliance ledger that produces an immutable, audit-ready record of every safety-critical clinical procedure. That whole stack is too big for one semester, so the final project deliberately scopes down to the **technical-feasibility core**: can a classical ML pipeline, trained on public IMU sensor data, classify human activity at the accuracy threshold a healthcare safety system would demand? If yes, the bridge to clinical procedure verification becomes a question of data collection and protocol mapping, not algorithmic invention.

I worked the project as a complete supervised classification pipeline on the **UCI Human Activity Recognition** dataset. The dataset comes from Anguita et al. (2013) and contains 10,299 windowed sensor recordings from 30 volunteers wearing a smartphone at the waist while performing six daily activities — walking, walking upstairs, walking downstairs, sitting, standing, and laying down. Each recording is represented as a 561-dimensional feature vector (time-domain statistics, frequency-domain features from FFT, inter-axis correlations) extracted from the smartphone's accelerometer and gyroscope at 50 Hz. The data arrived pre-split into 7,352 training samples and 2,947 test samples, with the split made by *subject* — 21 volunteers in train, 9 in test — so the test set is genuinely unseen.

The clinical mapping is explicit and limited: WALKING stands in for healthcare-worker movement between rooms, SITTING and STANDING for stationary procedural work like medication preparation or charting, and the transitions between activities for task initiation and completion. This is a deliberate simplification. Real handwashing has a specific rubbing-motion signature; real medication administration has a reach-scan-inject pattern; the surgical timeout has a specific clustering of multiple workers in close proximity for sixty seconds or more. None of those have public datasets, so UCI HAR is the stand-in I used to demonstrate that the underlying technology works.

The pipeline itself walks through the same workflow I'd practiced across Modules 3 through 9: data loading and quality checks, exploratory analysis, train/test split (honoring the pre-supplied subject-aware split, with 5-fold cross-validation on the training set for hyperparameter tuning so the test set was touched only once at the end), and four trained models compared head to head. The four models were chosen to span the bias-variance spectrum the course had taught me to think in:

- **Logistic Regression** as the linear baseline — would tell me whether the problem was even hard.
- **K-Nearest Neighbors (k = 7)** as a distance-based contrast — useful because the features were pre-normalized, removing the main thing that usually breaks KNN.
- **Linear Support Vector Machine (C = 10)** as the historical strong baseline on UCI HAR.
- **Random Forest (n = 200)** as the ensemble — variance reduction on top of decision trees, plus free feature importances.

Evaluation went beyond accuracy. I ran per-class precision, recall, and F1; built confusion matrices; pulled feature importance from the Random Forest; and specifically analyzed the *dangerous* confusions — the ones that would matter in a clinical setting, like predicting WALKING when the worker was actually SITTING (a "rounding" vs "stationary" mistake that translates directly to a patient-safety risk).

## Results

All four models cleared the 90% accuracy floor that the project's success criterion set. The performance ordering matched the bias-variance theory I'd learned in Modules 8 and 9 — the simplest model performed worst, and the variance-reducing ensemble performed best.

| Model | Test accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Logistic Regression | 87.5% | 0.865 | Below the clinical 90% floor — useful as a baseline only |
| KNN (k = 7) | 90.2% | 0.895 | Just clears the floor |
| Linear SVM (C = 10) | 92.1% | 0.915 | Comfortable margin above threshold |
| **Random Forest (n = 200)** | **94.8%** | **0.943** | Final selected model |
| UCI published benchmark | 89–96% | — | Anguita et al. (2013) and follow-ons |

The Random Forest landed at the upper end of the published UCI HAR benchmark band. Per-class behavior was lopsided in a way that turned out to be useful: static activities (SITTING, STANDING, LAYING) were essentially perfect with recall above 0.97, while the harder confusion was between WALKING_UPSTAIRS and WALKING_DOWNSTAIRS at around 0.92 recall each. In the clinical translation, that confusion is mostly harmless — both classes still indicate a healthcare worker moving between rooms. The dangerous confusion would be misclassifying WALKING as SITTING, and on the test set that happened in fewer than 1% of cases, comfortably under the 5% false-negative ceiling the success criterion set.

The Random Forest's feature importance ranking told a clean story too. The top-15 features were dominated by gravity-axis components and frequency-domain energies in the body-acceleration channels — in plain English, "is the body vertical or horizontal?" and "is the body shaking rhythmically?" Those two cues are exactly what would distinguish, say, scrubbing-at-a-sink from charting-at-a-station in the eventual clinical version.

## What I Learned

The biggest takeaway from CLINICPROOF was that **scope discipline matters more than ambition**. The abstract describes a multi-system platform with wearables, environmental sensors, ML, and a blockchain ledger; the final project deliberately solved only the ML feasibility piece, and admitting that openly produced a much stronger report than claiming to have built the whole platform would have. There's a real lesson there for any future industry project: the size of the vision and the size of the deliverable are different things, and the gap between them is where credibility either gets built or destroyed.

A few more specific things stuck with me:

- **The bias-variance arc the course teaches isn't abstract — it's a literal sequence I watched play out across four models on the same data.** Logistic Regression was below the clinical floor at 87.5%. KNN cleared it. SVM gave a comfortable margin. Random Forest delivered the headline number. Each step bought real accuracy for a real reason that maps onto Module 8's bias-variance framework and Module 9's variance-reduction story. The course material genuinely taught me how to *predict* the ordering before running the models, and seeing that prediction validated turned the theory from a vocabulary into an instinct.
- **Accuracy is the lazy metric.** A 94.8% headline number tells you almost nothing about whether a model is safe to deploy. The per-class recalls, the confusion matrix, and especially the dangerous-confusion analysis (WALKING vs SITTING under 1%) told me far more about clinical safety than the headline ever could. Going forward, the classification report is the first thing I'll read on any new model; accuracy is just the executive summary.
- **Subject-aware splits are the difference between honest and dishonest evaluation.** UCI HAR's pre-supplied train/test split puts 21 subjects in training and 9 in testing — those 9 subjects are people the model has never seen. If I'd done a random split instead, the model would have memorized individual gait patterns from training and posted a much higher (and meaningless) test accuracy. The honest measure of generalization to new healthcare workers is per-subject performance, and that single design choice in the dataset is what makes the 94.8% number trustworthy.
- **Feature importance is what turns a black box into a story you can tell.** The Random Forest's ranking — gravity-axis components and frequency-domain energies dominating the top 15 features — was something I could explain in plain English to a non-ML audience. That matters in healthcare specifically. Auditors, regulators, and clinicians will want to know *why* a model made the call it made, and a model whose top features map cleanly onto physical intuition (orientation and rhythm) is much easier to defend than a model whose decision boundary is a hairball.
- **Connecting the project back to the course modules made the whole thing coherent.** Module 5's data preparation showed up in the quality checks and the careful handling of the pre-normalized features. Module 8's bias-variance arc showed up in the model selection. Module 9's ensemble work showed up in why Random Forest beat the single-tree behavior the lab had warned me about. Module 10's PCA/clustering work didn't make it directly into this project, but it taught me to look at structure in data — a habit I leaned on during the EDA phase. The final project isn't a separate thing from the modules; it's the place where every module gets cashed in.
- **Real-world ML is mostly NOT the modeling step.** I spent more time on data quality checks, deciding which split strategy to honor, formatting confusion matrices so the dangerous-confusion analysis was easy to read, and writing the report than I did on actually fitting models. The four `model.fit()` calls combined took less than two minutes of compute. Everything else around them — that was the project.

## Challenges I Faced

- **Naming the gap between the vision and the deliverable honestly.** The CLINICPROOF abstract makes claims about hand-hygiene compliance, surgical safety checklists, fall prevention, and medication administration. None of those are something I actually built. The temptation was to pretend the UCI HAR pipeline was somehow a partial implementation of that whole vision. The truthful version is that I built a feasibility-check ML pipeline whose scope is "can sensor-based classification clear the 90% accuracy floor?" and that the answer being yes is a starting condition for the larger system, not a fragment of it. Saying that openly in the report was harder than dressing it up.
- **The dataset being a stand-in.** Every claim about clinical readiness in this project is conditional on a future data-collection effort, because UCI HAR is healthy volunteers performing scripted activities, not nurses in PPE working around patients. The single largest source of uncertainty in the final report is the gap between UCI HAR and a real hospital floor, and there's no way to close that gap inside a one-semester course. Naming it as the most important next step (rather than burying it in a footnote) was the right call but it felt like undermining my own headline number.
- **Choosing not to use deep learning.** Recent papers on UCI HAR using 1D CNNs and LSTMs trained on raw signals push accuracy into the 96–97% range. I deliberately stayed with classical models. The reasoning is real — clinical-domain priorities favor explainable, auditable models, and Random Forest's free feature importances buy more clinical credibility than the last 1–2 percentage points of accuracy a black-box network would have added — but defending that choice in the report meant pushing back on what looks like the obvious "best" answer, and getting that defense right took several drafts.
- **The 561-feature vector is opaque on its own.** Feature names like `tBodyAcc-mean()-X` and `fBodyAccJerk-bandsEnergy()-1,8` don't mean much without spending time with the dataset documentation. Trusting the engineered features rather than understanding each of them individually was a leap that felt uncomfortable at first. The feature-importance ranking from Random Forest helped — once I could see that the top features clustered into "orientation" and "rhythm" interpretations, the whole representation made sense — but until that point it was opaque.
- **Calibrating the dangerous-confusion analysis.** I had to invent the framing of "which off-diagonal cells in the confusion matrix would actually hurt a patient if this model were deployed?" — the standard ML metrics don't naturally surface that. Walking misclassified as walking-upstairs is harmless. Walking misclassified as sitting is the safety-critical mistake. Drawing that distinction took deliberate thinking, and the answer wasn't in any module of the course. It came from the abstract's healthcare framing colliding with the model's confusion patterns.
- **The lack of a real-world latency measurement.** The PRD specified <200ms inference latency as a clinical-readiness threshold, and the report records inference times measured under Colab on the full feature vector. But the clinical pipeline would have to compute those features from raw IMU streams on edge hardware first, and I had no way to measure that end-to-end. Recording the limitation honestly rather than hand-waving past it was the correct move, but it left a real gap in the feasibility claim.
- **Distinctness from FIELDPROOF.** My midterm project, FIELDPROOF, used the same general approach — body-worn sensor data, supervised classification — but for oil-and-gas field operations. The course guideline said the midterm and final topics should be sufficiently distinct, and I had to prove that explicitly: different domain, different dataset, different evaluation framing, different success criterion. Sharing a technical lineage is fine, but the projects had to land in clearly different places, and I built the appendix in the topic-selection submission specifically to handle that comparison head-on.

## Tools and Versions Used

- Python 3.11 in Google Colab
- pandas, NumPy, matplotlib, seaborn
- scikit-learn (`LogisticRegression`, `KNeighborsClassifier`, `LinearSVC`, `RandomForestClassifier`, `train_test_split`, `cross_val_score`, `accuracy_score`, `classification_report`, `confusion_matrix`, `recall_score`)
- UCI Human Activity Recognition Using Smartphones dataset (Anguita et al., 2013), source: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- The Joint Commission, *National Patient Safety Goals* (referenced for clinical context)

## What I Would Improve in Future Work

The five things on my list, in priority order:

1. **Collect real clinical data, even at small scale.** A pilot with even ten nurses doing handwashing for a week would convert the feasibility argument into a deployment argument. This is the single largest source of uncertainty and the one that pays off most.
2. **Move from windowed features to sequential models.** Procedure compliance is fundamentally about *order* — handwashing has eight sub-steps that must happen in sequence. A 1D CNN or LSTM on raw 50 Hz signals can learn that ordering directly, where the engineered-feature approach cannot.
3. **Run leave-one-subject-out cross-validation.** The pre-supplied UCI split is subject-aware, but a full LOSO-CV would give per-subject performance variance — the honest measure of generalization to new healthcare workers.
4. **Add the multi-sensor fusion the abstract describes.** The win for clinical compliance isn't better IMU classification alone; it's fusing IMU with BLE location, NFC interaction, and RFID equipment data so the model verifies *the procedure*, not just *the motion*.
5. **Measure end-to-end latency on edge hardware.** The clinical claim of <200ms response needs to be validated on the hardware that will actually run the model — not under Colab on pre-computed features.

## Acknowledgments

This project sits on top of two foundational works that deserve direct credit. **Anguita et al. (2013)** built and released the UCI HAR dataset, which is the entire empirical basis for the project. **The Joint Commission's** ongoing reporting on hand-hygiene and surgical-safety compliance documents the human-verification ceiling that automated systems like CLINICPROOF would need to break — it's the clinical motivation for the work, even if the final project doesn't use any of their data directly.


