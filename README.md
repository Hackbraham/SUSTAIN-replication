# SUSTAIN-replication
A remake of the **SUSTAIN** model created in https://www.gureckislab.org/publications/love_etal_2004.pdf as a final project 
for UW's Machine Learning course. 

Done in collaboration with [Aleksander Kałuski](https://github.com/aleksKaluski).

If you don't know what SUSTAIN is and you would like to use it, here is the summary of what SUSTAIN is:
* Adaptive architecture, increasing complexity to suit the problem 
* Unification of Supervised and Unsupervised Learning 
* Rule-based and Similarity-based behavior according to the problem 
* Attention emerges from the learning process 
* Order-dependence – prior models assumed presentation order didn’t matter 
* One of the first classification models with neurological grounding 
* One of the first formalizations of the parsimony principle that informs AI today

## Project outline
The classical problem of various classification models can be summarized as _the complexity dilemma_ (CD).

**CD**: _Simple models cannot map nonlinear labels, while complex models 
may perform poorly on simple problems._

SUSTAIN, which is an abbreviation for _Supervised and Unsupervised STratified Adaptive Incremental Network_, 
is essentially the response to CD that solves the problem by taking an inspiration from how people conceptualize categories.

### How do humans conceptualize categories?
1. **Human categories are correlated**, since some of them naturally co-occur. For instance, cats are included within
a set of animals with whiskers, but also a with animals with paws (so do other animals). In the most extreme examples,
some categories are co-extentional, meaning they relate to exactly the same objects (contingently), such as a category
of "animal with a heart" and "animals with a kidneys". 
2. **Humans organize categories hierarchically.** We use taxologies, such as _a cat is a mammal, which is an animal_.
3. **Human categories are multifunctional and relational.** Knowing what is a cat is knowing that is a mammal, and that it 
eats mouses and that there are many different cats.
4. **Human categories involve abstract concepts.** Cat is a mammal, but fish is not a mammal, but a whale…

### How's SUSTAIN addressing these problems?
1. Since the first part of learning is **unsupervised**; it clusters inputs before any feedback is given,
which means that it detects co-occurring features. 
2. S stands for **Stratified** - SUSTAIN it can represent different levels of a hierarchy by relying on different cluster granularities. 
3. The supervised top layer can then learn multiple distinct tasks on top of those same clusters, which implements **multifunctionality**.
4. Because SUSTAIN can be **supervised**, it implements abstract concepts.

### Model workflow
1. SUSTAIN stores categories as **clusters** (prototype-like summaries), not rules or individual examples. Starts with one; adds more as needed.
2. When a new item arrives, every cluster is **compared** to it by similarity. The closest cluster "wins."
3. The **winning cluster** generates a category prediction.
4. If the prediction is **wrong**, a new cluster is created centered on that item. If **correct**, the winning cluster is simply strengthened.
5. Across trials, SUSTAIN up-weights features that reliably predict category membership and down-weights irrelevant ones.

### Model's detailed description
SUSTAIN represents categories as a set of clusters, or recruited units, that grow incrementally rather than being fixed in advance.
Learning begins with a single cluster centered on the first item encountered, biasing the model toward the simplest possible
account of a category's structure. As new items arrive, each cluster's activation is determined by the (attention-weighted)
similarity between the item and the cluster's position, and these clusters compete so that the most active one tends to 
dominate the representation of a given item, in the spirit of a "winner take most" rather than a fully distributed code. 
The model only recruits an additional cluster when its current set of clusters cannot adequately handle an item, that is, 
when it encounters a surprising event such as being told a bat is a mammal rather than a bird under supervised learning; the 
new cluster is centered on that surprising item and becomes available to explain subsequent events, allowing the network's complexity 
to track the complexity of the category structure rather than overfitting from the start.

A selective attention mechanism works alongside this clustering process, weighting stimulus dimensions unevenly so the 
network focuses on whichever dimensions are most diagnostic for the current task, which further pushes the model toward 
parsimonious solutions. These attentional weights, along with the associations linking clusters to output values, are tuned 
by error-driven (gradient descent) learning, but only on the dimension the model is currently asked to predict. This is a query-based 
scheme that lets the same mechanism support different learning paradigms: predicting a category label yields ordinary supervised 
classification, predicting a missing feature yields inference learning, and predicting an item's own features with no externally 
supplied label yields unsupervised learning. Because cluster recruitment, attention allocation, and associative learning are all 
sensitive to the task and to the feedback the model receives, the substructure SUSTAIN discovers within a category is shaped jointly 
by the statistical structure of the stimuli and by the goals and demands of the particular learning task, which is how the model accounts 
for phenomena ranging from prototype-like abstraction to exemplar-like sensitivity to exceptions within a single architecture.

## Technical project description
### Structure
* `notebooks` - a folder with experiment and actual usage of SUSTAIN
  * `datasets` - data for experiments
  * `sustain_mushrooms.ipynb` - test of SUSTAIN on mushroom dataset
* `src` - source code of the model
  * `sustain.py` - the core of the model functionalities
* `replications.py` - replications of various classical experiments

## SUSTAIN in experiments
We conducted a series of experiments whose aim was to test the model's performance in 
a set of artificial scenarios implemented in the original paper as well as working 
with real datasets.

### Shepard's experiment
##### Stimuli
Eight stimuli varying along three binary perceptual dimensions plus a
binary category label (Shepard, Hovland & Jenkins, 1961). The same eight
items are split into two categories in six different ways, producing the
six classification problem types of increasing structural difficulty
(I through VI).

##### Experiment
Supervised classification. On every trial the label is queried and
corrective feedback is given. A learning block is one random pass through
all eight items; the criterion is four consecutive perfect blocks (max 32
blocks). Best-fitting parameters from Table 1 ("Six types") of Love et
al. (2004) are used.

##### Results
The expected human ordering is `I < II < III ≈ IV ≈ V < VI`. Our
replication recovers the qualitative ordering, with magnitudes slightly
inflated relative to the paper:
```pycon
Type I:   2.9 blocks    (~3 expected)
Type II:  10.1 blocks   (low expected)
Type III: 17.4 blocks   (~12 expected)
Type IV:  16.5 blocks   (~12 expected)
Type V:   20.8 blocks   (~12 expected)
Type VI:  27.1 blocks   (~16-20 expected)
```
SUSTAIN recruits ≈2.4 clusters for Type I and ≈8 for Type VI, matching
the paper's account of complexity scaling with category structure.

### Medin, Dewey & Murphy (1983)
##### Stimuli
Nine stimuli on four binary perceptual dimensions plus a "distinctive"
dimension that takes a unique value for each item (the idiosyncratic
photographic detail). Two conditions: **first-name** (nine unique labels,
i.e. identification learning) and **last-name** (two labels, i.e.
categorisation learning).

##### Experiment
Supervised classification, max 16 blocks, criterion of 2 consecutive
perfect blocks. The distinctive dimension receives an elevated initial
tuning `λ_distinct = 4.62` (Table 1, "First/last name").

##### Results
The paper reports the counter-intuitive finding that
identification (first-name) is faster than categorisation (last-name)
when stimuli are distinctive:
```pycon
                   humans   paper-SUSTAIN   ours
first_name (id):    7.1         7.2         6.2
last_name  (cat):   9.7         9.7         3.4
```
Our first-name number lands on target, but our last-name converges much
faster than the paper, reversing the qualitative pattern (see
`DISCREPANCIES.md`).

### Yamauchi & Markman (1998) / Yamauchi et al. (2002)
##### Stimuli
Four binary perceptual dimensions + a binary category label. Two category
structures: a **linear** structure (eight items, prototype-separable;
Yamauchi & Markman 1998, Table 5) and a **nonlinear** structure (six
items; Yamauchi et al. 2002, Table 6).

##### Experiment
Two tasks per structure:
* **Classification** — category label queried, perceptual dims given.
* **Inference** — category label given, one perceptual dim queried.

Each block presents every stimulus exactly once (per Yamauchi & Markman
1998 p. 69: "Each stimulus appeared once in each block"). For inference
the queried dim is a random *non-exception* perceptual dim — the
exception feature (the dim where the exemplar differs from its category
prototype) is excluded. Max 30 blocks; criterion is ≥ 90% mean accuracy
across three consecutive blocks. The category-label dim receives an
elevated initial tuning `λ_label = 5.15` (Table 1, "Infer./class.").

##### Results
```pycon
                            humans   paper-SUSTAIN   argmax    stochastic
linear    inference          6.5          7.5         4.9       30.0 (ceiling)
linear    classification    12.3         11.2         1.9        6.4
nonlinear inference         27.4         28.6         7.7       29.5
nonlinear classification    10.4         10.6         2.1        7.3
```
With argmax scoring, linear inference is accurate. With stochastic
sampling, nonlinear inference matches the paper closely. The "right"
scoring rule for SUSTAIN is left underspecified by both papers
(see `DISCREPANCIES.md`).

### Billman & Knutson (1996) Experiments 2 & 3
##### Stimuli
Seven ternary perceptual dimensions per item. Two correlation
structures per experiment:
* **Nonintercorrelated** — independent pairwise correlations (e.g.
  d1=d2; d3=d4; d5=d6).
* **Intercorrelated** — a single block of jointly correlated dims (e.g.
  d1=d2=d3=d4).

##### Experiment
Fully unsupervised: the model studies items for four blocks without
feedback, recruiting clusters whenever activation drops below
`τ = 0.5` (Eq. 11). At test the model is given 45 forced-choice pairs.
Following Billman & Knutson (1996, p. 463 and Table 2)'s "missing-parts"
procedure, each test pair has a *target rule* (a pair of correlated
dims) and the distractor mispairs those two values. The two blanked
dims are not random — they are the *other potentially informative
attributes* (the dims that share other correlations with the target
rule), so the model cannot use a different correlation to make the
judgment. Response probability is computed via Eq. 8 across the two
items' `C^out` on the (unitary) category-label output unit.

##### Results
```pycon
                                humans   paper-SUSTAIN   ours
Exp 2 nonintercorrelated:        0.62        0.66        0.62
Exp 2 intercorrelated:           0.73        0.78        0.69
Exp 3 nonintercorrelated:        0.66        0.60        0.59
Exp 3 intercorrelated:           0.77        0.78        0.70
```
All four conditions match the paper qualitatively (intercorrelated
> nonintercorrelated) and within ~10% quantitatively. 

### Real datasets
#### Mushrooms
##### Dataset
The mushroom [dataset](https://archive.ics.uci.edu/dataset/73/mushroom) is from UCI Machine
Learning Repository and describes hypothetical samples that correspond to the mushrooms 
from the _Agaricus_ and _Lepiota_ family. 

There are 8124 observations with 22 features such as _bruises, odor or gill size_.
The one that is most interesting for us is _poisonous_.

##### Encoding
We encoded the categorial featues of mushrooms into vectors by utilizing
`OrdinalEncoder` from `sklearn`. Thus, 22 features were respectively represented 
by 22 dimensions of a vector and each number of a given dimension characterized
a certain feature. 

```pycon
[2, 6, 4, 10, 2, 8, 2, 2, 2, 11, 2, 5, 4, 4, 8, 8, 1, 4, 3, 5, 7, 6, 7]
```
For instance, _edible_ and _poisonous_ labels are located as first dimension (so
the dimension = 2).

##### Experiment
We run 100 trial runs of SUSTAIN each time sampling 1000 rows from the dataset.
In each turn we used **supervised** mode of learning by splitting the dataset into 
train set (70%) and test set (30%) then forcing re-initialized SUSTAIN model 
to predict if an observation is poisonous. 

#### Results
We computed the classical evaluation metrics for ML models: accuracy, recall,
precision and f1-score. At the beginning our results were surprisingly 
good. 
```pycon
Mean Accuracy: 0.99
Accuracy SD: 0.02
Mean Precision: 0.99
Precision SD: 0.01
Mean Recall: 0.98
Recall SD: 0.04
Mean F1: 0.98
F1 SD: 0.03
```
The model performed excellent. Very high accuracy means that the model has enough data to learn the relations
between the features of mushrooms and predict the categories correctly. Another important point was the fact that
standard deviations was very low, which indicates that the model's performance was stable independently of the 
order of presentation. 

Was it really so stable? In order to test it, we ploted the results. We have to remember that although the data is 
plotted as line, the trials are independent! We discovered a fascinating phenomenon: around trial no. 73 SUSTAIN
has a massive downgrade of performance!

![SUSTAIN performance.png](photos/SUSTAIN%20performance.png)





## SUSTAIN's limitations


### N.B. 
This project was originally created without any AI use and then refined with Claude Code. 
