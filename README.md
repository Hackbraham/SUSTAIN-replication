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
#TODO

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





## SUSTAIN's liminations


### N.B. 
This project was originally created without any AI use and then refined with Claude Code. 
