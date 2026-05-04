# Module 2 Lab — Tools Used in Machine Learning

This repository contains my completed work for the **Module 2 Lab Exercise: Tools Used in Machine Learning**. The goal of the lab was to set up a working machine learning environment, get comfortable with the core Python libraries used throughout the course, load a real dataset, produce a first visualization, and document everything in a way that would be readable to someone reviewing the work later.

## Contents

- `Module_02_Lab_Exercise-1.ipynb` — the completed Jupyter notebook with all code cells executed and the reflection written.
- `README.md` — this file.

## What I Did in This Lab

The lab was structured in five parts. I worked through each one in order inside a single Jupyter notebook.

In **Part 1**, I set up my development environment. The lab gave three options — Google Colab, a local Jupyter installation, and VS Code with the Python and Jupyter extensions — and I confirmed the environment by running the first code cell, which imports pandas, NumPy, matplotlib, and scikit-learn. The cell printed the pandas and NumPy versions to confirm everything was wired up correctly.

In **Part 2**, I loaded the Iris dataset using `sklearn.datasets.load_iris()` and converted it into a pandas DataFrame. The dataset is the classic 150-row Iris flower dataset with four features — sepal length, sepal width, petal length, and petal width — and three target classes: setosa, versicolor, and virginica. I used `df.head()` and `df.info()` to inspect the structure and confirm there were no missing values.

In **Part 3**, I created my first visualization: a matplotlib scatter plot of sepal length versus sepal width, colored by species. I used a Python dictionary to map each species to a color (red for setosa, blue for versicolor, green for virginica) and looped over the unique species to plot each group separately so the legend would render correctly.

In **Part 4**, I practiced basic data operations. I used `groupby('species').mean()` to compute the mean of every numeric feature for each of the three species, and `value_counts()` to confirm the dataset is balanced — 50 samples per class.

In **Part 5**, I read through the lab's notes on GitHub workflow (creating a repository, committing, pushing) and on documenting projects with Markdown. I then completed the two assessment tasks at the end of the notebook:
- **Task 1** — computed the mean and standard deviation of sepal length using NumPy. Result: mean = 5.84 cm, standard deviation = 0.83 cm. The cell included assertion checks that confirmed the values were valid floats.
- **Task 2** — built a bar chart showing the count of samples for each species, confirming visually that the dataset is balanced (50 / 50 / 50).

Finally, I filled out the reflection cell with my observations about the dataset.

## What I Learned

The biggest takeaway from this lab was that the Python data-science stack is meant to be used together, not as a list of separate tools. Pandas does the data handling, NumPy does the math underneath, matplotlib does the plotting, and scikit-learn provides the dataset and (later) the algorithms. Each one has a specific job, and they hand data back and forth cleanly through DataFrames and NumPy arrays.

I also learned a few specific things that I'll keep using in later modules:

- **The Jupyter notebook workflow.** Being able to write a paragraph of explanation, run a code cell right below it, and see the output and any plots immediately changes how I think about exploring data. It's much closer to taking notes than to traditional programming.
- **What "loading a dataset" actually means in scikit-learn.** `load_iris()` doesn't return a DataFrame — it returns a Bunch object with `.data`, `.target`, `.feature_names`, and `.target_names` attributes. Wrapping it in a pandas DataFrame and adding a `species` column was the step that made everything afterwards (filtering, grouping, plotting) easy.
- **Why visualization comes before modeling.** The scatter plot showed me something I would not have seen from numbers alone: setosa is completely separate from the other two species in sepal-space, but versicolor and virginica overlap. That observation tells me ahead of time that any classifier using only sepal dimensions will struggle on those two species — and that's a useful thing to know before training a model rather than after.
- **The dataset is balanced.** Fifty samples per species means I do not need to worry about class imbalance for the Iris dataset, which is one less thing to think about when modeling. The lab pointed this out, and the bar chart in Task 2 confirmed it.
- **Documentation belongs next to the code.** Writing the reflection cell forced me to actually look at my own scatter plot and put what I saw into words. That habit of pairing every analysis with a short written summary is something I want to carry forward.

## Challenges I Faced

A few things tripped me up while working through the lab.

- **Choosing where to run the notebook.** I had three options — Colab, local Jupyter, VS Code — and spent more time than I expected just deciding. Colab was the path of least resistance because the libraries are pre-installed, so I went with that for this lab; on a local install I would have had to run `pip install` for the four libraries first.
- **Getting the import line right.** It is easy to mistype `from sklearn import datasets` or to import scikit-learn as `sklearn` and then forget the dot-separated submodule path. The standard aliases (`pd`, `np`, `plt`) feel arbitrary at first, but I learned quickly that it's worth using them exactly as everyone else does, because every example online assumes them.
- **Plotting multiple groups in one matplotlib chart.** My first instinct was to call `plt.scatter()` once with the whole DataFrame, but that produces a single-color cloud. I had to restructure the code to loop over the three species and call `plt.scatter()` once per species so that each group got its own color and a legend entry. This is a small thing, but it was the first place I had to actually *think* about how matplotlib works rather than copying a one-liner.
- **Reading the DataFrame output without getting lost.** `df.info()` prints column names, non-null counts, and dtypes; `df.head()` prints the first five rows. The two outputs look similar at a glance, and I had to slow down to actually understand what each one was telling me.
- **The Markdown reflection cell.** Writing prose about a dataset is harder than it sounds. The temptation is to restate what the numbers say ("there are 50 samples per species"). The harder, more useful thing is to say what the numbers *imply* ("the dataset is balanced, so I don't need to weight classes during training"). Pushing myself to write the second kind of sentence was a real exercise.

## Tools and Versions Used

- Python 3.13
- pandas 3.0.2
- NumPy 2.4.4
- matplotlib (default Colab version)
- scikit-learn (default Colab version)
- Jupyter Notebook / Google Colab

## Next Steps

Module 3 moves from *tools* into *types of machine learning* and asks me to build my first classifier. Based on what the scatter plot showed in this lab, my expectation is that a simple model will get setosa right easily and will need help separating versicolor from virginica — likely by using petal length and petal width instead of, or in addition to, the sepal dimensions I plotted here. That's the first thing I want to test in the next lab.
