# Learned Query Routing

## 1. Methodology

### 1.1 Motivation and Constraints

In a tiered retrieval system, the goal of query routing is to determine whether a query can be answered sufficiently using a smaller Tier-1 index or whether it should fall through to additional tiers. While this problem can naturally be framed as supervised classification, no ground-truth routing labels exist in practice.

Moreover, the MS-MARCO training relevance data presents an additional challenge. The available qrels are extremely sparse, often containing only a single judged relevant document per query. This sparsity makes traditional per-query effectiveness metrics unreliable as direct labeling signals.

These constraints strongly influenced our labeling strategy and ultimately motivated a rule-based approach grounded in judged relevance rather than metric comparisons.

### 1.2 Initial Consideration: Metric-Based Labeling (Abandoned)

Our initial approach was to define routing labels based on per-query effectiveness differences between Tier-1 and full retrieval. For example, if Tier-1 achieved MRR@10 or Recall@100 within some small $\varepsilon$ of the full system, the query would be routed to Tier-1 only, otherwise, it would fall through.

However, with only one (or very few) judged documents per query, such metrics become unstable. MRR@10 effectively collapses to a binary signal, while Recall@100 becomes dominated by the presence or absence of a single document. In this regime, small rank differences can cause large metric swings that are unrelated to true query difficulty.

As a result, metric-based labeling was deemed too noisy to produce reliable supervision under sparse relevance judgments.

### 1.3 Primary Labeling Rule: Judged Hit in Top-K

To address qrels sparsity, we adopted a hit-based labeling strategy grounded directly in judged relevance.

For each training query, we retrieve $\text{top-}k$ documents using:

- Tier-1 only retrieval
- Full retrieval (Tier-1 and Tier-2)

Labels are then assigned as follows:

- **Label 0 (Tier-1 sufficient)**: at least one judged relevant document appears in the Tier-1 $\text{top-}k$.
- **Label 1 (Fall through)**: no judged relevant document appears in Tier-1 $\text{top-}k$, but at least one appears in the full $\text{top-}k$.
- **Dropped**: no judged relevant document appears even in the full $\text{top-}k$.

This rule has several important properties. It is robust to sparse judgments, requiring only a single judged hit. It directly encodes the routing decision we care about (whether Tier-1 already retrieves something relevant) and avoids reliance on unstable per-query metric estimates.

This hit-based rule forms the core supervision signal for query routing.

### 1.4 Secondary Constraint: Pseudo-Recall Thresholding

While the hit-based rule is robust, it can be overly permissive. A query may retrieve a single judged document in Tier-1 while missing many other potentially relevant documents that are retrieved by the full system.

To address this, we introduce a pseudo-recall constraint as a secondary condition. For queries where Tier-1 retrieves at least one judged document, we compute:

$$
\text{pseudo-recall} =
\frac{\left| \text{Tier-1 top-}k \cap \text{Full top-}k \right|}
     {\left| \text{Full top-}k \right|}
$$

Here, the full system’s $\text{top-}k$ is treated as a proxy for the set of potentially relevant documents.

A query is labeled as Tier-1 sufficient only if both conditions hold:

1. A judged hit appears in Tier-1.
2. The pseudo-recall exceeds a threshold $t$.

If the pseudo-recall falls below $t$, the query is labeled as fall-through, even though Tier-1 retrieved a judged document.

This design penalizes cases where Tier-1 retrieves a judged hit but diverges substantially from full retrieval, while also introducing a tunable parameter that allows controlled exploration of effectiveness and efficiency trade-offs.

### 1.5 Thresholds and Label Variants

Rather than selecting a single pseudo-recall threshold, we train separate routing models using thresholds:

$$
t \in \{0.0, 0.1, 0.2, \dots, 0.9\}
$$

The case $t = 0.0$ corresponds to the pure hit-based rule, where no pseudo-recall constraint is enforced. Higher thresholds encourage increasingly conservative routing, requiring Tier-1 to recover a larger fraction of the documents retrieved by the full system.

Training multiple models allows us to study routing behavior across operating points, avoid baking a single policy into the model, and evaluate routing sensitivity without retraining indexes. In effect, the threshold becomes a tunable control knob for routing aggressiveness.

### 1.6 Query Features

We restrict ourselves to query-only features, ensuring that routing decisions can be made before retrieval begins.

The features fall into two categories.

**Structural query features** capture query verbosity and redundancy:

- Number of terms
- Number of characters
- Number of unique terms
- Fraction of unique terms
- Average term length

**Collection-level IDF statistics** capture term selectivity:

- Maximum IDF
- Minimum IDF
- Mean IDF
- Standard deviation of IDF

Intuitively, short queries with rare, high-IDF terms are more likely to be answered sufficiently by a smaller index, whereas longer or more generic queries benefit from broader coverage. All features are computable without accessing postings lists, making them efficient and deployable in a latency-critical routing component.

Importantly, feature definitions are fixed and identical across all models, ensuring that differences in routing behavior arise solely from thresholding and training data, not feature drift.

### 1.7 Model Choice

We use a lightweight linear classifier (logistic regression) for routing.

The routing decision is binary and low-dimensional, making linear models a natural fit. Logistic regression is fast, interpretable, and easy to deploy, which is important for a component that sits on the critical path of query processing.

More complex models such as gradient boosting were considered but rejected to avoid unnecessary complexity and latency. Training separate classifiers for each threshold allows the decision boundary to adapt naturally to different routing aggressiveness levels, rather than forcing a single model to approximate multiple operating points.

## 2. Implementation

### 2.1 Index and Dataset Partitioning

During the index tiering phase of the project, the MS-MARCO document collection was split into:

- `collection_train.tsv`, which was used exclusively to train the index tiering model.
- `collection_work.tsv`, which contains all documents referenced by evaluation qrels plus additional randomly sampled documents.

All tiered indexes used for query routing are built from `collection_work.tsv`, which represents the deployed system’s document universe. This separation ensures that index tiering decisions are learned from training data, while routing decisions operate on the same indexes used at evaluation time.

### 2.2 Training Data Selection

To train the query routing model, we start from MS-MARCO’s `queries.train.tsv` and `qrels.train.tsv`. Because the corpus was artificially partitioned, not all judged documents appear in the working collection. We therefore filter training queries to those whose judged documents exist in `collection_work.tsv`.

To keep retrieval tractable, we randomly sample 5,000 queries from the filtered set, a strategy approved by the course instructor.

### 2.3 Training Pipeline

For each sampled training query, we:

1. Retrieve $\text{top-}k$ using Tier-1.
2. Retrieve $\text{top-}k$ using full retrieval.
3. Generate labels using the hit-based rule and pseudo-recall threshold.
4. Extract query features.
5. Repeat for each threshold $t$.

This produces 10 labeled datasets, one per threshold, from which we train 10 independent routing models.

### 2.4 Inference and Evaluation

At inference time, each query is featurized and passed through a selected routing model, which predicts whether to remain in Tier-1 or fall through. Retrieval is then executed accordingly using the tiered indexes.

We evaluate routed retrieval on eval1 and eval2 using all thresholds, and on the dev set using a subset of thresholds due to its larger size. Metrics are reported both overall and stratified by query length.
