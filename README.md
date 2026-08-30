# Bayesian Optimisation with Gaussian Processes

## Documentation Links

* 📄 **[Model Card](MODEL_CARD.md)** - Details on model architecture, evaluation metrics, and intended use.
* 📊 **[Dataset Datasheet](MODEL_DATASHEET.md)** - Overview of training data, collection process.
* 📄 **[Model Evolution]( MODEL_EVOLUTION.md)**  - Detailed breakdown of model additions for every optimisation round

---

## Project Overview

The capstone project centres around a black box optimisation challenge where the aim is to maximise the output of 8 unknown functions. The functions simulate a range of real world scenarios and vary in dimensionality from 2D to 8D. Only a limited number of initial observations is available for each function and the query budget is fixed to 12 evaluations. 

The objective is to develop an appropriate search methodology fitting to multi-dimensional sparse sets to efficiently and accurately locate the maximum of each function within the query budget.

My approach uses Bayesian optimisation with gaussian processes to fit a probabilistic surrogate model to existing observations. An acquisition function is then used to determine where to sample next by balancing exploration (regions where model uncertainty is high) and exploitation (regions close to good observations). Once a new candidate is selected, the true function is re-evaluated and the process is repeated. 
     
---

## Project Motivation

Many real world optimisation problems involve objective functions that are expensive to evaluate. Rather than evaluating the objective exhaustively, Bayesian Optimisation attempts to determine **which point should be evaluated next** based on information collected from previous evaluations.

Examples include:
- Hyperparameter optimisation
- Engineering design
- Experimental design
- Materials discovery
- Simulation-based optimisation

---

## Inputs and Outputs

The model built receives a set of initial observations (inputs and corresponding outputs) and uses it to suggest a new query point. Initial observation sample size varies per function.

### Model inputs:

2D NumPy array of observation inputs of size $(n_{samples}, n_{dims})$ 

${X} = [[x_1,x_2, …x_{dims}], [x_1,x_2, …x_{dims}], …]$

1D NumPy array of observation outputs of size $(n_{samples},)$ 

${y} = [y_1, y_2, … y_{samples}]$

<ins>Example of 1 sample of observation inputs and outputs for a 3D function:</ins>
- Input: [0.89, 0.81, 0.7]
- Output: [12.45]

### Model outputs:

A vector of size $(n_{dims})$ representing the next suggested query point.

${X} = [x_1,x_2, …x_{dims}]$

<ins>Example output query vector for a 3D function:</ins>
- Next query: [ 0.96, 0.05, 0.65 ]

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

### Current approach – Final model 

The full workflow of the final model including all available features is described below:

<ins>Data</ins> 

During every round, for each model run, the original observation data set is loaded and all new query points appended, creating a new set with all existing observations.

- Observations function input data is already scaled in the unit cube range requiring no further processing. 
- Observations function outputs are transformed where necessary to manage large range variation (Function 1 outputs are transformed into log space before the GP is fitted)
  
<ins>Sampling Grid</ins>

- Dense 2D meshgrid is used for generating the evaluation grid for the 2D functions
- Space filling Sobol sampling is used to generate the evaluation grid for the 3D-8D functions.

Even though the density of the evaluation grid defines search space coverage and is therefore important during optimisation, the BoTorch differentiable acquisition functions resample multiple times during gradient based surface optimisation reducing dependence on initial density while still providing excellent coverage.  

<ins>Gaussian Process Surrogate</ins>

The surrogate model is built using the BoTorch library methods. 

Depending on the function dimensionality a standard (2D – 5D) or sparse axis prior (>5D) is chosen and then the GP model is trained on the existing observations to produce the surrogate. 

The GP is using the Matern 5/2 kernel (chosen over RBF during hyperparameter tuning) as it strikes a better balance between smoothness and flexibility.  
Hyperparameter marginalisation is performed automatically on sparsity strength, dimension lengthscales, observation noise and other prior distribution variables via directly sampling from the posterior. This allows for assessment of multiple hyperparameter sets that are appropriately then combined to capture additional model parameter uncertainty used to better calibrate the model. 

<ins>Model calibration</ins>

After fitting, the model calibration (over or under confidence) is assessed via cross validation techniques. Leave-one-out is used for smaller observation sets (Functions 1-6) and the k-fold option used for larger observation sets (Functions 7,8).

If cross-validation shows heavy mis-calibration, the model can be refitted by applying user defined bounds on lengthscales and noise (This is a selectable option by passing additional arguments and not the default).  

<ins>Acquisition Functions</ins>

Next candidate suggestions are generated using the following acquisition function implementations already included in BoTorch:

- Upper Confidence Bound (qUpperConfidenceBound)
- Expected Improvement (qLogExpectedImprovement)
- Probability of Improvement (qProbabilityOfImprovement)
- Thompson Sampling

Each acquisition function trades off exploration and exploitation differently, which can allow for investigation of promising regions or search of uncertain areas in the design space. As a result, candidates from each acquisition can lead to very different optimisation trajectories. 

<ins>Exploration vs Exploitation</ins>

The current model does not impose an automated exploration vs exploitation strategy as rounds progress. It instead summarises suggested candidates from all acquisition functions providing an extended amount of metrics, enabling the user to make a final choice.

- UCB is the only function that has a manual parameter passed to the model allowing explicit control over exploration and exploitation ( This parameter is gradually reduced as the rounds progress). 
- EI and PI are controlled by the value passed as the reference target and the exploration/expoitation trade-off is indirect. 
- Thompson sampling is driven by the stochastic properties of the GP posterior with the trade-off handled purely probabilistically hence the exploration, exploitation balance is self-regulating and adaptively driven by posterior variance.

A trust region and exclusion zone features are included to expand user control over exploration and exploitation. The trust region bounds the search within a user defined zone of the search space. The exclusion zone uses minimum distance to force next queries away from existing observations.

<ins>Candidate selection</ins>

Reviewing a combination of manually induced and self-regulating trade-offs from different acquisitions can inform next query decisions. When candidate suggestions agree, confidence in predicted performance of a particular region increases. When candidates disagree understanding of the reasons can drive exploration of alternative regions.

A candidate comparison report is generated including:

- Next candidate coordinates, predicted posterior mean and standard deviation
- Distances between candidates and between candidates and best observation or minimum distance to any of the existing observations
- An acquisition cross-evaluation table normalised by acquisition score showing how candidates suggested from one acquisition perform is the subspaces of other acquisition functions

<ins>Neural Network</ins>

A neural network surrogate model has been implemented and trained on existing observations to provide a separate independent candidate prediction that could be compared with the GP.  This method served only as an additional estimate due to concerns around surrogate reliability being poor given the sparsity of observations and was only used for a few rounds. GP remained the main predictive model used throughout the optimisation.

<ins>Reporting and visualisation</ins>

An extensive number of metrics is being reviewed and reported throughout the optimisation along with multiple visualisations to provide further insight into model behaviour, data trends and candidate predictions. These include:

- Cross validation metrics on model fit
- Lengthscale metrics (mean and standard deviation) based on posterior samples
- Candidate comparison reports
- Principal component decomposition into function dimensions and reporting on high variance directions
- 2D Slice contour plots of posterior mean, standard deviation and acquisition score, overlayed by existing observations and highlighted best and next queries.
- Parallel coordinate plots showing variability in input dimensions across all observations contoured via function output
- 2D PCA projection plots of observations, incumbent best and next candidate suggested points along the two principal components overlayed by biplot arrows showing how each function dimension drives the function output.
- Optimisation progression plots tracking best incumbent variation over all optimisation rounds
- Lengthscale magnitude progression plots over all optimisation rounds

---

## Results

### Optimisation Performance

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

