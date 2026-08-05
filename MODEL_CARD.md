# Model Card

## Model Overview

Model name: Black Box Bayesian Optimiser

Version: v1.0

Model Type: Sequential Bayesian optimisation using gaussian process regressors 

---

## Intended Use

### Primary Use Cases

The model is well-suited for machine learning professionals and general users wishing to apply ML for optimisation of black box functions where function evaluation is expensive and only a small number of initial observations is available. 

Some use case examples include:

* Hyperparameter tuning
* Engineering design optimisation via e.g simulation or low order model data 
* Experimental data optimisation

### Out-of-scope

Model use should be avoided:

* Where highly automated, parallelised workflows are required, as currently next suggested candidate final selection requires manual input. 
* The SAAS option should only be used for high dimensional functions where most inputs are expected to have minimal impact on the output value.

--- 

## Model details and Strategy

The model evolved throughout the duration of the project. Early rounds used simple single kernel Scikit learn implementation of Bayesian optimisation with the UCB acquisition function. In later rounds, more focus was spent on improving model fit and design space search by advancing to the BOtorch library and introducing:

* Hyperparameter marginalisation
* Proper probabilistic priors as well as sparse dataset specific priors for high dimensional functions such as SAAS
* Multiple acquisition functions such as UCB, EI and Thompson sampling including advanced acquisition functions within BOtorch using Markov Chain Monte Carlo sampling.
* Candidate selection tables summarising key metrics and performance across acquisition functions
* More sophisticated visualisation techniques such as dimension parallel plots and contour slices

A neural network model was created half way through the project to provide independent candidate suggestions. However the Gaussian Process model remained the main method used for query submissions.

--- 

## Performance

The main objective was to maximise each function. The global maximum was unknown hence any improvement over the initial observation dataset best point is considered beneficial.

Overall, improvement has been recorded across all functions. Consistency in identification of good candidate points improved during later rounds where more data was available.

The main metric used to assess performance post – candidate submission was increase in function output over best initial observation. 

Candidate selection however, was based on a combination of posterior mean and standard deviation, predicted improvement, the exploration/exploitation settings and agreement between acquisition function suggestions in terms of location (distance between different acquisition candidates and between acquisition candidates and best observation).

--- 

## Assumptions and limitations

### Assumptions

* Black box functions are continuous and sufficiently smooth
* Most dimensions are weakly correlated to the function output (When SAAS prior is used)

### Limitations

* Computational expense is high for large datasets. Significant slowdown expected for datasets with >1000 observations.
* Exploration/Exploitation acquisition function parameters and transition are problem specific and should be revised when applied to a new problem/function

---

## Ethical Considerations

The black box function data do not contain any sensitive or personal information. 

Transparency in the context of the current project is maintained via clear documentation of the model assumptions, limitations and overall methodology and approach to the optimisation, such that results are reproducible. 

This further supports and guides responsible usage of the developed framework in new real-world problems and applications. 
