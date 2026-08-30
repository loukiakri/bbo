# Method Evolution

## Round 1 – Initial Gaussian process setup

The first round focused on implementing an initial Bayesian optimisation workflow using Scikit-Learn by setting up:

* Gaussian process surrogate fitting
* Sobol grid sampling
* Acquisition functions (UCB)

At this stage 2 kernels were tested, Matern and RBF and hyperparameters were varied manually for both kernels and the acquisition function to understand the impact on surrogate fit and predictions. 

## Round 2 – Setup improvement

The second round focused on improvement of the original workflow by relaxing fixed kernel assumptions to allow higher flexibility within the surrogate fit, exploring more of the available hyperparameters and expanding the overall toolset.

Main changes:

* Addition of ARD functionality on lengthscales, allowing variation in lengthscale per dimension and setting of appropriate bounds
* Addition of kernel scaling and noise parameters that are also optimised along with the lengthscales via the kernel internal optimiser, tracking log likelihood. 
* Addition of EI and Thompson sampling acquisition functions

Thompson sampling was specifically added  to capitalise on the inherent randomness in sampling of the posterior, making it harder to get stuck in a local optimum and thereby more robust when used on the multimodal functions. 

## Round 3 – Understanding model reliability

The third round focused on understanding how to ensure the GP fit is reliable to better inform the acquisition functions.

Main changes:

* Implementation of an ensemble method, running multiple kernel model refits over a resampled dataset and combining the mean and standard deviation.
* Implementation of cross-validation (loop-one-out and k-fold) for assessing model over/under fitting.

The ensemble was introduced such that disagreement between fits could be used as an additional uncertainty with regards to the variability of observation points, with the resampling used to assess robustness of the fit to points in certain dimensions.
In retrospect this methodology was not suitable for the small datasets available, as any re-sampling or loss of even one independent point could lead to massive instability in the fit that would get transferred into the uncertainty of the posterior. As a result all of the queries produced during this round yielded very suboptimal results. This method was removed in future iterations of the code.

## Round 4 – BoTorch switch

During the fourth round the Bayesian optimisation workflow switched from scikit-learn to the BoTorch library, taking advantage of the fully Bayesian inference options available and proven to perform well in budget constrained, sparse set problems, such as the capstone.

Main changes:

* Full code refactoring using the BoTorch library methods
* Implementation of hyperparameter marginalisation, proper prior distributions, differentiable acquisition functions
The main motivation behind the switch was based on the following three key aspects of the BoTorch implementation:

**Hyperparameter marginalisation:** The BOtorch fully Bayesian models use advanced techniques to draw many plausible hyperparameter sets from the posterior and average predictions over all of them. This propagates hyperparameter uncertainty into every prediction on top of model uncertainty and is absolutely critical for high dimensional sparse sets where hyperparameters are under-determined. The scikit learn simple GP, keeps only a single best guess hyperparameter set ignoring hyperparameter uncertainty and thereby being more prone to overconfidence. This is very important for the optimisation evolution since the acquisition functions trade-off exploiting high predicted values and exploring high uncertainty regions. If the uncertainty is estimated as too small (scikit learn GP) the acquisition under-explores and stops sampling regions that should be investigated. 

**Priors:** In scikit learn hyperparameters are usually constrained by user defined ranges. In BoTorch priors can be applied as probability distributions. For small data sets this helps better regularise the fit towards sensible hyperparameter values when the data can’t describe/pin them down fully. In addition, the availability of sparsity specific priors for lengthscale such as SAAS in BoTorch (Assuming only a few dimensions matter in high dimensional spaces) helps impose structure on the lengthscales for e.g. functions 7 and 8, where due to limited budget and sparsity would have been difficult to learn from the data.

**Differentiable acquisition functions:** In scikit learn acquisition functions are typically evaluated on a predefined grid or random points set with the maximum value taken forward. As dimensions increase the evaluation grid size becomes a constraint and the acquisition is never really optimised, only sampled. BoTorch acquisition functions are differentiable and gradient based optimisation is applied on their surface to find genuine optima. In high dimensions this is advantageous as there is less dependence on the evaluation grid and better coverage leading to identification of potentially better optima.

## Round 5 – Code restructuring and Neural Network surrogate

The fifth round focused on inclusion of a neural network surrogate and general improvements to the code structure. 

Main changes:

* Implementation of a neural network model surrogate used in conjunction with the acquisition functions providing independent candidate predictions
* Code restructuring and organisation into models, plotting and utilities re-usable modules rather than a single notebook

## Round 6 – Hyperparameter tuning

The sixth round focused on investigation and tuning of the BoTorch kernel hyperparameters. During the solution there are certain hyperparameters that are varied and sampled from the posterior such as lengthscales and noise. However, the BoTorch implementation includes other hyperparameters that tend to remain fixed along all runs relating to the priors and the posterior sampling, directly affecting the model fit.  

Main activities:

* Investigated SAAS prior shrinkage scale defining the level of strength of imposed sparsity as well as other standard prior distribution options affecting the predictive uncertainty in the posterior or the assumed noise levels.
* Experimented with warm_up steps, num_samples and thinning all of which govern how the hyperparameter posterior distribution is explored by defining how many samples are eventually used for marginalisation.
* Investigated the gradient based acquisition function parameters such as num_restarts and raw_samples governing optimiser accuracy and convergence
  
Appropriate levels for each hyperparameter were specified and their effect on surrogate fit and candidate predictions assessed. The tuned hyperparameter version of the model was then used to produce next round predictions.

## Round 7 – Code enhancements and candidate scoring

The seventh round focused on refining the GP implementation and introducing candidate metrics reporting

Main changes:

* Implementation of additional controls to limit the BoTorch prior ranges on noise and lengthscales for cases where cross validation showed miscalibration. Made these parameters user selectable for both standard and sparse axis (SAAS) priors used in the high dimensional functions.
* Creation of a compare_candidates.py module for performing candidate comparison including:
** Comparison table summarising next candidate suggestions along with predicted posterior mean and standard deviation for different candidates, reported alongside incumbent best.
** Reporting of distance metrics between different candidate suggestions and amongst existing observations and best observation
** Acquisition function cross evaluation scoring table of suggested candidates (from EI, UCB, PI, Thompson sampling) based on acquisition score.

The candidate metrics report improved confidence in candidate selection by providing further insight into the agreement on the location of a high performing basin amongst acquisition functions with different exploration/exploitation strategies or e.g  by highlighting evidence of clustering via the distances reported.

## Round 8  – Improving code readability and computational efficiency 

The eighth round was spent improving computational efficiency of the BoTorch implementation and overall code readability. With an increasing number of points the code was becoming extremely slow during next candidate evaluations, often crashing and leading to memory allocation issues (BoTorch fully bayesian methods are much more computationally expensive than simple scikit-learn fits).

Main activities:

* Added  batch size and sample size parameters as model inputs to overcome memory errors.  These control the maximum batch evaluation chunk size passed to PyTorch during sampling and optimiser evaluation. 
* Re-organised the code, removed repeated functionality within different scripts and  further improved readability, report print-outs and general commenting. 

Both code streamlining and the memory fix improved execution speed without affecting prediction performance. 

## Rounds 9 – 10  – In depth analysis of optimisation progression 

During these rounds, no new algorithms were introduced with the focus being in understanding the trends in the data based on the optimisation progression so far and adjustment of the exploration/exploitation strategy. Both reporting and visualisations were extended to further support this analysis.

Main activities:

* Created a detailed lengthscale report showing mean and standard deviation that can also be tracked over different optimisation rounds to visualise any shifts in lengthscale magnitude or variability based on the posterior samples. This was initiated as Function 2 data indicated a potential second high performing basin which manifested as next candidate suggestion disagreement between acquisitions only on the long lenthscale dimension.
* Refined existing 2D contour slice plots of posterior mean, standard deviation and acquisition score overlayed by existing observations and highlighted best observation and next suggested candidate.
* Introduced an optimisation progression plot capturing changes in incumbent best over all rounds
* Improved existing parallel coordinate plots used for efficient visualisation of the differences in input dimensions between points, particularly for high dimensional functions by adding contouring of each observation line by the posterior mean.
* Reviewed candidate reports and plots extensively identifying both over-exploration (e.g. Function 1)  and over-exploitation (e.g  Function 8) trends. 

Both the improved plotting and detailed report reviews highlighted important data trends that informed the adjustment of the exploration/exploitation trade-off for the different functions in the next rounds making most efficient use of the remaining evaluation budget.

## Round 11 – Trust region and exclusion zone

The eleventh round focused on more aggressively correcting the over-exploration and over-exploitation trends observed in previous rounds rather than just relying on the acquisition parameter tuning.

Main changes:

* Addition of a trust region zone defined by the user. The zone bounds next candidate search only to a specific section of the domain such that the user can isolate a promising region and refine the search within it.
* Implementation of an exclusion zone using minimum distance from existing observations to avoid next candidate suggestions that are duplicate (or near duplicate)  and control the level of clustering around best performing points.

## Round 12 – PCA Diagnostics
The twelfth round focused on inclusion of PCA diagnostics and plots to further analyse the design space and provide insight on the model predictions and existing observations.

Main changes:

* Created 2D plots projecting observations and next suggested candidates from different acquisitions into PCA space (2 principal components) contoured by function output.
* Added scree plot visual and diagnostics showing split of variance per principal component as well as breakdown of function dimension contribution to each principal component.
* Added a highlight feature for top performing observations
* Added Bi-plot arrows to indicate the function dimensions correlation to the principal components

The PCA plots further highlighted suspected clustering and over-exploitation in certain functions. Understanding of the composition of each principal component was also insightful when comparing to the lengthscale metrics particularly for the high dimensional functions. 

## Round 13 – Final submission

This round captured the final query submission. No further adjustments were made to the code, instead all the existing functionality was methodically used to inform decisions and maximise the function output results. The optimisation strategy shifted to direct exploitation across all functions.

Candidates from all acquisitions were still generated with final selection being biased towards the PI and UCB (with very low beta) candidates that exhibited the highest posterior mean and lowest uncertainty predictions. 

---

All of the above led to the development of a sophisticated robust approach that is adaptable to the optimisation needs and has the right tools to enable thoughtful iteration and informed decision making. 


