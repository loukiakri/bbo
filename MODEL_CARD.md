# Model Card

## Model Overview

**Model name:** Black Box Bayesian Optimiser

**Version:** v1.0

**Model Type:** Sequential Bayesian optimisation using gaussian process regressors 

**Model Inputs:** Multi-dimensional NumPy arrays capturing initial observations and evaluated query outputs

**Model Outputs:** NumPy array with the input dimensions of the next query candidate suggestion 

**Model Architecture:** BoTorch implementation of Bayesian optimisation with Gaussian Processes for single objective optimisation using standard and sparse axis priors and differentiable acquisition functions.

---

## Intended Use

### Primary Use Cases

The model is well-suited for machine learning professionals and general users wishing to apply ML for optimisation of black box functions where function evaluation is expensive and only a small number of initial observations is available. 

Some use case examples include:

* Hyperparameter tuning
* Engineering design optimisation via e.g simulation or low order model data 
* Experimental data optimisation
* Materials discovery

### Out-of-scope

Model use should be avoided:

* Where highly automated, parallelised workflows are required, as currently next suggested candidate final selection requires manual input. 
* The SAAS option should only be used for high dimensional functions where most inputs are expected to have minimal impact on the output value.

--- 

## Model details and Strategy

The model evolved throughout the duration of the project. Early rounds used simple single kernel Sci-kit learn implementation of Bayesian optimisation with the UCB acquisition function. In later rounds, more focus was spent on improving model fit and design space search by advancing to the BoTorch library and introducing:

* Hyperparameter marginalisation
* Proper probabilistic priors as well as sparse dataset specific priors for high dimensional functions such as SAAS
* Multiple acquisition functions such as UCB, EI, PI and Thompson sampling that are differentiable and can be gradient-based optimised.
* Cross-validation (Loop-one-out and k-fold) techniques and emtrics for assessing model over/under confidence
* Candidate selection tables summarising key metrics and performance across acquisition functions
* More sophisticated visualisation techniques such as dimension parallel plots, 2D contour slices of posterior mean, standard deviation and acquisition score as well as PCA space projections.
* User defined trust region and exclusion zones for additional control of exploration/exploitation and local refinement

A neural network model was created half way through the project to provide independent candidate suggestions. However, the Gaussian Process model remained the main method used for query submissions.

--- 

## Performance

The main objective was to maximise each function. The global maximum was unknown hence any improvement over the initial observation dataset best point is considered beneficial.

Overall, improvement has been recorded across all functions as can be seen in the summary table below. Consistency in identification of good candidate points improved during later rounds where more data was available and the underlying model was improved and further calibrated.

| Function | Dimensions | Initial Observations Best | Optimisation Best | Query Round of Best | Inputs @ Best |
|---|---:|---:|---:|---:|---|
| F1 | 2D | `7.710875e-16` |`6.884216e-09` |10| `[0.723435, 0.678126]` |
| F2 | 2D | `6.112052e-01` |`7.344425e-01` |12| `[0.695816, 0.273372]` |
| F3 | 3D | `-3.483531e-02` |`-8.296981e-03` |13| `[0.350469, 0.714455, 0.41873]` |
| F4 | 4D | `-4.025542` |`6.239079e-01` |7| `[0.418607, 0.404969, 0.426197, 0.405082]` |
| F5 | 4D | `1.088860e+03` |`8.662405e+03` |6| `[0.999999, 0.999999, 0.999999, 0.999999]` |
| F6 | 5D | `-7.142649e-01` |`-1.261755e-01` |13| `[0.390069, 0.353407, 0.691056, 0.742125, 0.111512]` |
| F7 | 6D | `1.364968` |`2.769198` |13| `[0.248004, 0.075584, 0.220421, 0.305477, 0.347138, 0.714349]` |
| F8 | 8D | `9.598482` |`9.991800` |13| `[0.083747, 0.16894 , 0.162633, 0.164701, 0.809849, 0.457978, 0.22954 , 0.658543]` |

The main metric used to assess overall performance was increase in function output over best initial observation as well as subsequent improvement round by round.

Candidate selection however, was based on a combination of metrics including:

* Predicted posterior mean and standard deviation
* Acquisition scores from different functions such as EI/UCB/PI/Thompson sampling predicting improvement
* The exploration/exploitation settings and phase of the optimisation
* Agreement/Disagreemnt between acquisition function suggestions in terms of location of next suggested point
* Distance metrics between different acquisition candidates and between acquisition candidates and best observation
* Acquisiton candidate cross-evaluation scores

Cross-validation metrics showing model over/under confidence were also taken into account in every optimisation round to adjust model expectation and assess model reliability.

--- 

## Assumptions and limitations

### Assumptions

* Black box functions are continuous and sufficiently smooth. If the functions being modelled are suspected to have sharp discontinuities, the GP can predict artificial oscillations near abrupt changes and inflate uncertainty away from transitions driving the acquisition function into suboptimal spaces.
* Most dimensions are weakly correlated to the function output (When SAAS prior is used). If more function dimensions strongly impact function output and the SAAS prior is used, possible underfitting occurs in the active subspace which can prevent true optimum identification/miss design space patterns.

### Limitations

* Computational expense is high for large datasets. Significant slowdown expected for datasets with >1000 observations. Even at datasets of ~100 observations, high posterior sampling parameters can overflow local memory and should be adjusted based on computational resources.
* The default model setup can process moderate homegeneous noise however alternative priors and settings should be used for extremely noisy functions or functions where noise varies spatially within the domain.
* Exploration/Exploitation acquisition function parameters and transition are problem specific and should be revised when applied to a new problem/function

---

## Ethical Considerations

The black box function data do not contain any sensitive or personal information. 

Transparency in the context of the current project is maintained via clear documentation of the model assumptions, limitations and overall methodology and approach to the optimisation, such that results are reproducible. 

This further supports and guides responsible usage of the developed framework in new real-world problems and applications. 
