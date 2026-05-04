# Module 10 Lab — Unsupervised Learning

This repository contains my completed work for the **Module 10 Lab: Unsupervised Learning**. Every previous lab in this course — Modules 2 through 9 — has been about supervised learning, where the dataset comes with both features and a target label and the model's job is to learn the mapping between them. Module 10 is the first time the target column disappears entirely. That's a fundamental shift in what "learning" even means: instead of telling the algorithm what the right answer is, you ask it to find structure on its own.

## Contents

- `Module_10_Lab_-_Unsupervised_Learning.ipynb` — the completed Jupyter notebook with all four code cells executed and the Knowledge Check answered.
- `README.md` — this file.

## What I Did in This Lab

The lab covered the two most important unsupervised learning techniques: **K-Means Clustering** (for finding groups in data) and **Principal Component Analysis** (for compressing high-dimensional data into fewer dimensions). It was organized into three parts plus a Knowledge Check.

In **Part 1 — Supervised vs. Unsupervised Learning**, I read through the conceptual setup. Supervised learning has features `X` and labels `y`; unsupervised learning has only `X`. The two main families of unsupervised algorithms are **clustering** (grouping similar data points together) and **dimensionality reduction** (compressing data into fewer features without losing much information).

In **Part 2 — K-Means Clustering**, I generated a synthetic dataset of 300 customer-spending-style data points using `make_blobs` with 4 true centers and `cluster_std=0.8`. The lab framed this as a customer segmentation problem — pretending I didn't know in advance that there were 4 groups, and using the data to discover that.

I then completed two K-Means tasks:

- **Task 1 — The Elbow Method.** I ran K-Means with `k` from 1 to 10, recorded the **inertia** (sum of squared distances from each point to its assigned cluster center) for each, and plotted the result. The plot showed inertia falling steeply from k=1 to k=4 and then leveling off — a classic "elbow" shape, with the bend at k=4 confirming that the data has 4 natural groups.
- **Task 2 — Clustering and visualization.** I trained `KMeans(n_clusters=4, random_state=42, n_init=10)` on the same data, extracted the cluster labels via `kmeans.labels_` and the cluster centers via `kmeans.cluster_centers_`, and plotted everything: 300 points colored by their assigned cluster, with the four centroids overlaid as red X markers. The four clusters in the plot matched the four blobs that `make_blobs` had originally generated, even though K-Means had no access to the true labels — it had recovered the structure on its own.

In **Part 3 — Principal Component Analysis**, I switched to the **Iris dataset** (the same one I had used in Module 9) and used PCA to compress its 4 features down to 2 principal components for visualization. The numbers came out cleanly:

- **Principal Component 1 explained 92.46% of the variance.**
- **Principal Component 2 explained 5.31% of the variance.**
- **Together, the two components captured 97.77% of all the variance in the original 4-dimensional dataset.**

I plotted the result as a 2D scatter colored by Iris species. Even though PCA itself is unsupervised — it had no idea which flowers were which species — the three species formed clearly separated groups in the new 2D space, with setosa cleanly isolated and versicolor and virginica adjacent but distinguishable. That separation is what makes the PCA plot the canonical "Iris is easy to classify" visualization.

In the **Knowledge Check**, I answered three questions: the fundamental difference between supervised and unsupervised learning (the presence or absence of labels); why we don't just maximize `k` to minimize inertia (because inertia goes to zero when every point becomes its own cluster, which destroys all structure rather than capturing it); and what the 97.77% explained variance result tells me about Iris (that the four original features are highly correlated, so the dataset is effectively near two-dimensional and can be analyzed in 2D with very little information loss).

## What I Learned

The biggest takeaway was that **unsupervised learning answers a different question than supervised learning** — and that the question changes everything about how you evaluate the result. In supervised learning, "good" is well-defined: the prediction matches the label, and accuracy / precision / recall measure how often. In unsupervised learning, there is no label to compare against. "Good" becomes a judgment call: do these clusters look meaningful? Does this lower-dimensional representation preserve enough structure? Is the elbow really at k=4 or could it be at k=3? Trading the comfort of a single accuracy number for the openness of "does this match what I see?" is a real cognitive shift.

A few specific things stuck with me:

- **Inertia always decreases with k, so it can't be the whole story.** The naive instinct is "I want low inertia → I'll just pick the highest k." But inertia goes to zero when every point is its own cluster, which is the opposite of useful. The elbow method exists precisely because the metric is monotone — what matters is not the metric itself but the *change* in it. The elbow is the point where adding another cluster stops paying for itself. That pattern of "the metric only tells half the story; you have to look at how it changes" is going to show up over and over in ML.
- **Synthetic data lets you check the algorithm's homework.** I used `make_blobs(centers=4)` to generate the data, then "forgot" that I knew there were 4 groups, and asked K-Means to find them. The fact that K-Means recovered exactly 4 clusters is not just confirmation that the algorithm worked — it's a sanity check that I trust. On a real dataset where I don't know the true number of clusters, I can't run that sanity check, so I'd have to lean harder on the elbow plot and on domain interpretation of what each cluster actually represents.
- **PCA is a coordinate system change, not a feature selection.** This was the key insight that made PCA click. The four original Iris features (sepal length, sepal width, petal length, petal width) don't disappear when you run PCA — they get *combined* into two new features called PC1 and PC2 that are weighted mixtures of all four originals. Each principal component is a direction in the original 4D space along which the data varies the most. Choosing 2 components doesn't drop 2 of the original 4 features; it builds 2 entirely new features that together preserve as much variance as possible.
- **97.77% variance explained means the data has redundancy.** The four Iris features are not independent measurements — they're correlated. Petal length and petal width move together; sepal length is partly correlated with both. The fact that 2 components capture 97.77% of the variance is the mathematical statement of "this dataset really only has 2 dimensions worth of information; the other 2 are mostly redundant." That's exactly the same insight Module 9 hinted at when feature importance showed petal width and petal length together carrying 85% of the discriminative power.
- **Unsupervised + supervised work together.** PCA reduced Iris from 4D to 2D without seeing the species labels. Then I plotted the 2D result colored *by* species, and the species fell into clean groups. That's a powerful workflow: use unsupervised learning to find structure, then check it against labels (when you have them) to confirm the structure is meaningful. In practice this pattern is how you decide whether a dimensionality reduction is worth keeping.
- **K-Means is sensitive to initialization.** The `n_init=10` parameter I passed to KMeans means "run the whole algorithm 10 times with different random starting centroids and pick the best result." Without that, K-Means could converge to a local minimum where one cluster captures two real groups and another cluster gets nothing. The parameter is small but consequential, and it's another instance of the pattern from earlier labs where a one-line setting prevents a real bug.

## Challenges I Faced

- **The shift from "predict" to "discover."** Every previous lab ended with an accuracy number. This one ended with a scatter plot and a judgment call. The first time the elbow plot came up I caught myself looking for "the answer" the way I would for a classification result — but the elbow method doesn't give you a single answer, it gives you a curve that you interpret. Getting comfortable with "looks like 4" instead of "is exactly 4" is its own skill.
- **Reading the elbow plot.** The bend at k=4 was visually clean on this synthetic dataset, but it's worth noting how lucky that is. On a noisier real dataset, the curve could bend gradually rather than sharply, and "where's the elbow?" could legitimately have two reasonable answers (k=4 or k=5, say). The elbow method is a heuristic, not a deterministic rule, and pretending otherwise sets you up for false confidence.
- **Understanding what `inertia_` actually measures.** It's the sum of squared distances from every point to *its assigned* cluster center — not the global mean. The first time I read the formula I had to slow down to check that. The "its assigned" part is what makes inertia decrease as `k` increases: more clusters means each point is closer to *its own* center, even if the centers themselves are not particularly meaningful.
- **Interpreting principal components.** PC1 explained 92.46% of the variance, but what *is* PC1? It's a weighted combination of the four original features — but the lab didn't have me extract the loadings (the weights) to see which features contributed most. Without that step, PC1 stays a black box: I know it captures most of the variance, but I don't know what it "represents." I want to come back to this and inspect `pca.components_` so I can give each principal component a meaningful interpretation.
- **Why PCA worked so well on Iris specifically.** Getting 97.77% variance explained from 2 components is an unusually good result — it's not what you'd get on most real-world datasets. Iris is small (150 samples), low-dimensional (4 features), and has highly correlated measurements. On a dataset like, say, 1000 image pixels, you might need 50+ principal components to capture the same amount of variance. That context matters for not over-generalizing from this lab.
- **The temptation to trust K-Means too much.** Because the synthetic data was generated with 4 true centers and K-Means recovered 4 clusters, the algorithm looked infallible. It's not. K-Means assumes spherical clusters of similar size, and breaks when those assumptions don't hold — for example, on data with elongated clusters, nested rings, or very different cluster densities. DBSCAN and Gaussian Mixture Models exist precisely to handle cases K-Means can't, and the lab didn't get into them. That's a good thing to remember before reaching for K-Means as a default.

## Results Summary

**Part 2 — K-Means on synthetic blob data (300 points, 4 true centers):**

| Step                  | Result                                                                  |
|-----------------------|-------------------------------------------------------------------------|
| Elbow method (k=1–10) | Clear elbow at **k=4**                                                  |
| K-Means(k=4)          | Recovered the 4 generating centers; cluster assignments matched true blobs |

**Part 3 — PCA on Iris (4 features → 2 components):**

| Component             | Variance explained |
|-----------------------|--------------------|
| Principal Component 1 | 92.46%             |
| Principal Component 2 | 5.31%              |
| **Total (2 components)** | **97.77%**       |

In the 2D PCA plot, the three Iris species formed clearly separable groups even though PCA had no access to the species labels.

## Tools and Versions Used

- Python 3
- pandas, NumPy, matplotlib, seaborn
- scikit-learn (`make_blobs`, `KMeans`, `PCA`, `load_iris`)
- Jupyter Notebook
- Synthetic blob dataset (300 samples, 4 centers, `cluster_std=0.8`, `random_state=42`)
- Iris dataset, loaded via `sklearn.datasets.load_iris()`

## Next Steps

Module 11 should move into more advanced unsupervised techniques or into the integration of unsupervised steps into a supervised workflow. Three things I want to take into it:

1. **Inspect the PCA loadings, not just the explained variance.** The `pca.components_` attribute gives me the weight of each original feature in each principal component. Looking at it would tell me whether PC1 is mostly "petal size" or whether it's a more diagonal mixture of all four features — and that interpretation is what turns a black-box compression into a meaningful re-representation of the data.
2. **Try clustering algorithms that don't assume spherical clusters.** DBSCAN and hierarchical clustering both relax the assumptions K-Means makes about cluster shape and size. Running them on the same blob data would replicate the K-Means result; running them on data with non-spherical clusters would expose the cases where K-Means fails. Either way, I'd build intuition for when to reach for what.
3. **Use PCA as a preprocessing step for a supervised model.** Compressing Iris to 2 components and then training a classifier on those two new features would let me ask: did dimensionality reduction help, hurt, or leave performance unchanged? On a near-2D dataset like Iris I'd expect it to leave performance roughly intact. On a high-dimensional dataset, it could meaningfully speed up training and reduce overfitting. That's the supervised-meets-unsupervised pipeline I want to build comfort with.


