# SUSTAIN-replication
A remake of the **SUSTAIN** model created in https://www.gureckislab.org/publications/love_etal_2004.pdf as a final project 
for UW's Machine Learning course. 

Done in collaboration with [Aleksander Kałuski](https://github.com/aleksKaluski).

## Project outline
The classical problem of various classification models can be summarized as _the complexity dilemma_ (CD).

**CD**: _Simple models cannot map nonlinear labels, while complex models 
may perform poorly on simple problems._

SUSTAIN, which is an abbreviation for _Supervised and Unsupervised STratified Adaptive Incremental Network_, 
is essentially the response to CD that solves the problem by taking an inspiration from how people conceptualize categories.

### How do we conceptualize categories?
1. **Human categories are correlated**, since some of them naturally co-occur. For instance, cats are included within
a set of animals with whiskers, but also a with animals with paws (so do other animals). In the most extreme examples,
some categories are co-extentional, meaning they relate to exactly the same objects (contingently), such as a category
of "animal with a heart" and "animals with a kidneys". 
2. **Humans organize categories hierarchically.** We use taxologies, such as _all animals are mammals_
3. **Human categories are multifunctional.**
4. **Human categories involve abstract concepts.**

## Technical project description
### Structure
* `notebooks` - a folder with experiment and actual usage of SUSTAIN
  * `datasets` - data for experiments
  * `sustain_mushrooms.ipynb` - test of SUSTAIN on mushroom dataset
* `src` - source code of the model
  * `expretimetnal_tools.py` - a set of tools useful for conducting simple experiments
  * `sustain.py` - the core of the model functionalities

### Model principles of operation
#TODO (stratified structure, adaptive recruitment, incremental learning, supervised 
vs unsupervised)

### Model workflow

## SUSTAIN in experiments
We conducted a series of experiments which aim was to test the model's performents in 
a set of artifical scenarios imlemented in Shepard's paper as well as while working 
with real datasets.

### Shepard's experiments
#TODO

### Real datasets
#### Mushrooms
#TODO

#### Another experiment
#TODO

## SUSTAIN's liminations


### N.B. 
This project was originally created without any AI use and then refined with Claude Code. 