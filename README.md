# Bayesian Optimisation with Gaussian Processes

## Documentation Links

* 📄 **[Model Card](MODEL_CARD.md)** - Details on model architecture, evaluation metrics, and intended use.
* 📊 **[Dataset Datasheet](MODEL_DATASHEET.md)** - Overview of training data, collection process.

---

## Project Overview

The capstone project centres around a black box optimisation challenge where the aim is to maximise the output of 8 unknown functions. The functions simulate a range of real world scenarios and vary in dimensionality from 2D to 8D. Only a limited number of initial observations is available for each function and the query budget is fixed to 12 evaluations. 

The objective is to develop an appropriate search methodology fitting to multi-dimensional sparse sets to efficiently and accurately locate the maximum of each function within the query budget.

My approach uses Bayesian optimisation with gaussian processes to fit a probabilistic surrogate model to existing observations. An acquisition function is then used to determine where to sample next by balancing exploration (regions where model uncertainty is high) and exploitation (regions close to good observations). Once a new candidate is selected, the true function is re-evaluated and the process is repeated. 

At a high level, the optimisation loop is:

```text
                 ┌───────────────────────┐
                 │  Initial observations │
                 └──────────┬────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Fit GP Surrogate  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Acquisition Function│
                 │    Optimisation     │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Select Candidate  │
                 └──────────┬──────────┘
                            │
                            ▼
                    Evaluate Candidate
                            │
                            └──────────────► Repeat
```

Achieving the project objective goes far beyond model selection. It requires building a robust and systematic method, tracking the right metrics and using thoughtful iteration based on practical data driven reasoning.

---

## Project Motivation

Many real world optimisation problems involve objective functions that are expensive to evaluate. Rather than evaluating the objective exhaustively, Bayesian Optimisation attempts to determine **which point should be evaluated next** based on information collected from previous evaluations.

Examples include:
- Hyperparameter optimisation
- Engineering design
- Experimental design
- Materials discovery
- Drug discovery
- Simulation-based optimisation

---

## Inputs and Outputs

The model built receives a set of initial observations (inputs and corresponding outputs) and uses it to suggest a new higher performing next query point. The functions vary in dimensionality from 2D to 8D  and so does the initial observation sample size.

Model inputs
* 2D numpy array of observation inputs of size (nsamples, ndims) 

$$ 
\mathcal{X} = [[x_1,x_2, …x_{dims}], [x_1,x_2, …x_{dims}], …]
$$

* 1D numpy array of observation outputs of size (nsamples)

$$
\mathcal{y} = [y_1, y_2, … y_{samples}] 
$$

Example of 1 sample of observation inputs and outputs for a 3D function:
Input  - [0.89, 0.81, 0.7]
Output - [12.45]

Model outputs:

A vector of size (ndims) representing the next suggested query point for each function.
Example output query vector for a 3D function:
Next query - $$ [0.96, 0.05, 0.65 ]$$

---

## Technical approach

### Overall Method

Bayesian optimisation with gaussian process regressors is the main method used to tackle the capstone project. It is extremely useful as instead of spending query budget and resources uniformly across the search space it instead attempts to use each evaluation strategically. The gaussian processes provide probabilistic predictions and explicitly estimate uncertainty which is critical in determining where additional information is more valuable and where to query next.

### Methodology evolution (Model development)

In the early rounds the Bayesian optimisation approach was mainly built with the Sci-kit learn library and it featured simple single kernel GP fits with fixed hyperparameter values for e.g lengthscale or noise. Only one acquisition function was used and the acquisition parameters were kept constant across all functions. 

The approach evolved significantly over the course of the project focusing on two main pillars: 
* Improving the surrogate model fit per function and building the right tools that could monitor and assess that while making adjustments for function dimensionality and sparsity.
* Creating a scoring framework for assessing multiple next candidate options and choosing the most appropriate one given the query budget and overall objective while balancing exploration and exploitation.

Given the constraints of the project (query budget) and the dataset (very few initial observations) it was evident that the methodology developed had to cater to sparse sets that vary in dimensionality. 
This realisation was the main reason/motivation for switching from the scikit-learn library to the BOtorch library enabling fully Bayesian inference, that has clear advantages on problems of this nature. 

A detailed model evolution description of the implementation changes from each round is available in the  📄 **[Model Evolution]( MODEL_EVOLUTION.md)**  document.

The final BoTorch based model includes:
-	Hyperparameter marginalisation
-	Proper priors (Distributions over hyperparameter values)
-	Differentiable acquisition functions that are gradient optimised
-	A library of acquisition functions EI, UCB, PI, Thompson sampling
-	Leave-One-Out and k-fold cross validation for assessing model fit
-	Capability of defining an exclusion zone or trust region via setting a minimum distance from existing points or area bounds
-	Advanced plotting techniques for visualising the design space such as 2D slice contour plots, Parallel coordinate plots and PCA projections
-	Next candidate reports detailing mean, std, distance from best/min distance, cross evaluation score

### Strategy (Exploration VS Exploitation)

---

## Installation

### Requirements

The project is built primarily using:
- Python
- PyTorch
- BoTorch
- GPyTorch
- NumPy
- SciPy
- Matplotlib
- Sci-kit learn

### Clone the Repository

```bash
git clone https://github.com/loukiakri/bbo.git
cd bbo
```
### Create a Virtual Environment

---
## Results

### Optimisation Performance

---

## Future Work

Potential extensions include:
