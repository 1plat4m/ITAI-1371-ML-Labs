# Module 4 Lab — Exploratory Data Analysis

This repository contains my completed work for the **Module 4 Lab: Exploratory Data Analysis**. Module 2 was about installing and importing tools. Module 3 was about running the full ML workflow on a clean dataset. Module 4 is a step backward in the pipeline and a step forward in skill: before you can train a model, you have to understand the data well enough to know whether the model has a chance. EDA is the work that happens between getting the data and writing the first `train_test_split` line.

## Contents

- `Module_04_Lab_-_Exploratory_Data_Analysis.ipynb` — the completed Jupyter notebook with all code cells executed and the Knowledge Check answered.
- `README.md` — this file.

## What I Did in This Lab

The lab used the **Titanic dataset** — 891 passengers with eleven columns including `Survived` (0 or 1), `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked`. The notebook was organized into four parts plus a Knowledge Check.

In **Part 1 — Setup and Data Loading**, I imported pandas, matplotlib, and seaborn, then loaded the Titanic CSV directly from the data-science-dojo GitHub repository. Calling `df.info()` immediately surfaced two things worth keeping in mind: the `Age` column has missing values (714 non-null out of 891), and the `Cabin` column is mostly missing. This is realistic data — unlike the Wine dataset in Module 3, it does not arrive ready to model.

In **Part 2 — Descriptive Statistics**, I ran `df.describe()` to get the summary table for the numeric columns and pulled out three headline numbers:
- The average passenger age was **29.7 years**.
- The overall survival rate was **38.4%** — meaning the dataset is imbalanced.
- Fares ranged from **$0** to **$512** — a huge spread that hinted at the wealth disparity I'd see later.

In **Part 3 — Visual EDA**, I built four guided plots, each with a stated insight:

1. **Survival distribution** (`countplot` of `Survived`) confirmed the imbalance: about 60% died, 40% survived.
2. **Survival by passenger class** (`countplot` of `Pclass` with `hue='Survived'`) showed that 1st-class passengers survived at a much higher rate than 3rd-class passengers — money helped.
3. **Survival by gender** (`countplot` of `Sex` with `hue='Survived'`) showed the strongest single pattern in the dataset: women survived at a much higher rate than men. The "women and children first" mantra was data, not folklore.
4. **Survival by age** (`FacetGrid` of `Age` histograms split by `Survived`) showed that non-survivors clustered in the 20–40 range, while survivors had a noticeable spike of young children.

In **Part 4 — Student Experimentation**, I built two of my own visualizations:

- **Experiment 1: Port of Embarkation.** I ran a `countplot` of `Embarked` split by `Survived` and then computed survival rates per port using `df.groupby('Embarked')['Survived'].mean()`. The numbers were:
  - **Cherbourg (C): 55.4%**
  - **Queenstown (Q): 39.0%**
  - **Southampton (S): 33.7%**

  Cherbourg's much higher survival rate is almost certainly a class effect — more 1st-class passengers boarded there.

- **Experiment 2: Fare vs. Survival.** I built a side-by-side `boxplot` and `violinplot` comparing fare distributions for those who died versus those who survived. The contrast was stark: median fare for those who died was **$10.50**, while median fare for those who survived was **$26.00** — almost double. This is a different angle on the same wealth signal that Pclass and Embarked surfaced.

In the **Knowledge Check**, I answered three questions: the primary goal of EDA (understanding the data's shape, missing values, and relationships before modeling); the profile of a Titanic survivor most likely to make it (a young, female, 1st-class child); and why visualization beats summary statistics alone (because two datasets can have identical means and standard deviations but completely different shapes — Anscombe's Quartet).

## What I Learned

The biggest takeaway was that **EDA is where the model gets designed**, even though no model is being trained yet. Every plot in this lab generated a candidate feature for a future classifier: `Sex`, `Pclass`, `Fare`, `Age` brackets, and `Embarked` all carry signal about survival. By the time I reached the end of Part 4, I already had a mental model of what a Titanic classifier needs to learn — and that's the whole point of EDA.

A few specific things stuck with me:

- **Three signals point to the same underlying cause.** `Pclass`, `Fare`, and `Embarked` all separate survivors from non-survivors, and all three are really measuring the same thing: wealth. That insight changes how I'd build a model. Throwing all three into a logistic regression would create correlated features that compete for the same explanatory weight; understanding that they're related lets me make better feature choices ahead of time.
- **Imbalanced datasets are the default, not the exception.** Wine in Module 3 was 71/59/48, which felt mildly uneven. Titanic is 549/342, which means a model that always predicts "died" would already be 61.6% accurate. That makes the 88.9% accuracy I got in Module 3 look much less impressive, and it explains why precision, recall, and F1 — not accuracy — are the metrics that matter on imbalanced data.
- **Categorical data needs categorical plots.** A `countplot` with a `hue` parameter does in one line what would take five or six lines of pandas grouping and bar-plotting in matplotlib alone. Knowing which seaborn function fits which data type is a genuine productivity skill.
- **Missing data is not just a cleaning problem.** The fact that `Age` is missing for 177 passengers is itself a clue. It's worth asking *why* the data is missing — were younger / wealthier / male passengers more likely to have their age recorded? Treating missingness as informative rather than just "a thing to drop" is the kind of EDA habit that changes downstream model quality.
- **Boxplots and violin plots tell different stories.** The boxplot collapsed the fare distribution to five numbers (min, Q1, median, Q3, max). The violin plot showed the actual *shape* — and the survivor distribution had a long tail toward higher fares that the boxplot only hinted at. Using both together is more useful than either alone.
- **Anscombe's Quartet is the answer to "why bother visualizing?"** Four datasets with identical means, variances, and correlation coefficients but completely different shapes when plotted. Once that example clicks, you stop trusting summary statistics on their own.

## Challenges I Faced

- **Reading `df.info()` carefully.** The output looks dense at first — a column name, a non-null count, and a dtype for every column. The trick is to scan the *non-null counts* first: every column with a count below 891 has missing data, and the size of the gap tells me how worried to be. `Age` (714 non-null) is fixable; `Cabin` (about 200 non-null) probably has to be dropped.
- **The percentage trick I almost missed.** The lab uses `df['Survived'].mean()` to get the survival rate, treating the 0/1 column as a proportion. It works because the mean of a 0/1 column equals the fraction of 1s. The same trick made my Embarked experiment trivial: `df.groupby('Embarked')['Survived'].mean()` gives survival rates per port directly. This pattern is going to be one of my most-used pandas idioms.
- **A real warning I had to investigate.** When I ran the Fare boxplot/violin experiment, seaborn printed a `UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.` The plots still rendered correctly, but the warning was real — modern matplotlib wants `ax.set_xticks([0, 1])` to be called before `ax.set_xticklabels(['Died', 'Survived'])` or wants the labels passed via `ax.set_xticks([0, 1], ['Died', 'Survived'])`. Knowing to read warnings as actionable feedback rather than noise is a habit I'm still building.
- **Choosing between boxplot and violin plot.** I initially didn't know which one to use for Experiment 2, so I built both side-by-side and compared. That's a reasonable pattern when you don't know — it costs almost nothing and you learn something either way — but I'd want to make a more informed choice on a real project.
- **Making sense of the Embarked result.** Cherbourg's 55.4% survival rate looked surprising at first. The instinct is to say "the port caused the difference," but that's almost certainly wrong — the port is a proxy for class. Distinguishing correlation from causation from a single plot is harder than the bar charts make it look, and it's the kind of thing that requires checking the cross-tab (Pclass by Embarked) before drawing conclusions.
- **The temptation to skip EDA and start modeling.** After Module 3 I wanted to jump straight to `train_test_split`. Forcing myself to spend a whole lab on plots without training a single model felt slow at first, but by Experiment 2 I understood why: I now have a much better sense of what features matter and which ones are correlated, which means any classifier I build later will start from a stronger foundation.

## Results Summary

| Question I asked the data | Method                              | Headline finding                                                |
|---------------------------|-------------------------------------|-----------------------------------------------------------------|
| How many survived?         | `countplot` of `Survived`           | 38.4% survived — imbalanced                                      |
| Did class matter?          | `countplot` of `Pclass` × `Survived` | Strong: 1st-class survived at much higher rates                  |
| Did gender matter?         | `countplot` of `Sex` × `Survived`   | Strongest single signal: women survived far more often           |
| Did age matter?            | `FacetGrid` histograms of `Age`     | Children spike in survivor distribution                          |
| Did port of boarding matter? | `groupby('Embarked')['Survived'].mean()` | Cherbourg 55.4% · Queenstown 39.0% · Southampton 33.7%        |
| Did fare matter?           | `boxplot` + `violinplot` of `Fare`  | Median fare $26 (survived) vs $10.50 (died)                      |

## Tools and Versions Used

- Python 3
- pandas
- matplotlib, seaborn
- Jupyter Notebook
- Titanic dataset, loaded from `https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`

## Next Steps

Module 5 should move into data cleaning and preparation, which is exactly the natural follow-on to this lab. Three things I want to take into it:

1. **Handle the missing `Age` values.** The lab surfaced the missingness but did not address it. I want to compare three strategies — drop those rows, fill with the median, or fill with a class- or gender-specific median — and see how the choice affects downstream model performance.
2. **Engineer a `Title` feature from the `Name` column.** Names like "Mr.", "Mrs.", "Miss.", and "Master." encode age and gender information that's already in other columns, but they also pick up status markers like "Dr." or "Rev." that aren't. This is the classic Titanic feature-engineering move and a good place to practice.
3. **Build the model EDA suggested.** Given everything this lab uncovered, a logistic regression on `Sex`, `Pclass`, `Age`, and `Fare` should land somewhere in the high 70s on accuracy. Confirming that prediction is the cleanest possible bridge from "I understand the data" to "I can model it."
