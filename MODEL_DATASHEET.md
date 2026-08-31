# Model Datasheet

## 1. Motivation

### Why was the dataset created?
The dataset was created to capture the observations used in the optimisation of 8 unknown black box functions as part of a black box optimisation (BBO) project. The project objective was to maximise each individual function under a limited query budget of 13 evaluations.

### What task does it support? 
The dataset supports Bayesian optimisation, surrogate modelling using gaussian process regressors, acquisition function tuning and optimisation and next candidate selection strategies for optimisation of black box functions.

---

## 2. Function Overview

All function descriptions, dimensionality and initial observation sample size, as provided for use in the capstone project, are summarised in the table below.

| Function | Dimensions | Initial Observation Number | Optimisation Goal | Function Description |
|---|---|---|---|---|
| F1 | 2D | 10 | Maximise | Detection of contamination sources in a two-dimensional area, such as a radiation field, where only proximity yields a non-zero reading. |
| F2 | 2D | 10 | Maximise | A mystery ML model, that takes two numbers as input and returns a log-likelihood score that needs to be maximised. Noisy outputs and many local peaks|
| F3 | 3D | 15 | Maximise | A drug discovery project, testing combinations of three compounds to create a new medicine. Goal is to minimise side effects by optimising a transformed output (e.g. the negative of side effects). |
| F4 | 4D | 30 | Maximise |Hyperparameter tuning for an ML model used for accelerating costly calculations related to optimally placing products across warehouses for a business with high online sales. Multiple local optima |
| F5 | 4D | 20 | Maximise | A four-variable black-box function that represents the yield of a chemical process. The function is unimodal and the single peak where yield is maximised should be identified |
| F6 | 5D | 20 | Maximise |A cake recipe is represented using a black-box function with five ingredient inputs that is evaluated by a final score where factor contributed negative points. Goal is to bring this function as close to zero as possible | 
| F7 | 6D | 30 | Maximise | Optimisation of an ML model by tuning six hyperparameters. The goal is to find the combination of hyperparameters that yields the highest possible performance| 
| F8 | 8D | 40 | Maximise | Optimisation of an eight-dimensional black-box function aiming to maximise function output |

## 3. Dataset Composition

The dataset consists of the initial observations prior to the start of the optimisation (for each function) and the round by round observations after each new function evaluation is performed.

### Contents

* **Total Size / Scale:**
The size and dimensionality of the original observation dataset is as recorded in the table above for each function. By the end of the optimisation the observation dataset size increases by 13 points for all functions (e.g Function 1 started with 10 initial observations and the final dataset after optimisation completion will contain 23 observations).
* **Data Types:**
Both inputs and outputs are continuous numerical values. Inputs range within the unit cube [0,1]^d across all functions whereas output range can be very different and is function specific spanning multiple orders of magnitude.
* **File Formats:**
All data points are stored in NumPy arrays and text files

### Observations

Observations consist of pairs of input and output variables. The input arrays are dependent on function dimensions.

Input: 2D NumPy array of observation inputs of size $(n_{samples}, n_{dims})$ 

${X} = [[x_1,x_2, …x_{dims}], [x_1,x_2, …x_{dims}], …]$

Output: 1D NumPy array of observation outputs of size $(n_{samples},)$ 

${y} = [y_1, y_2, … y_{samples}]$

<ins>Example of 1 sample of observation inputs and outputs for a 3D function:</ins>
- Input: [0.89, 0.81, 0.7]
- Output: [12.45]

### Gaps
The dataset is sparse especially for the high dimensional functions. There are no missing values or NaNs that need to be removed/imputated.

---

## 3. Collection

### Data Collection

Following the initial set of observations each new query was generated through the workflow described below:
* The gaussian process was used to create a surrogate model via training on existing observation data
* A new evaluation grid was created using Sobol sampling
* Different acquisition functions were evaluated with the surrogate model over the new grid
* Acquisition function evaluation output was used to provide a new candidate suggestion
* Selected candidate was submitted and true function value returned
* New observation was added to the existing dataset

The process was repeated in the next query round.

The timeframe of the project was 13 weeks. One evaluation per week for each function was permitted

### Strategy

The optimisation strategy evolved within the duration of the project.

Early rounds focused on exploration of the design space targeting reduction of overall uncertainty. Later rounds transitioned into exploitation of the promising regions that were being progressively identified. 

The transition point was different for each function depending on function dimensionality, the outputs of each round and model fit metrics. 

---

## 4. Preprocessing & Uses

### Pre-processing

* Inputs are already scaled to the unit cube. 
* Outputs were standardised to zero mean and unit variance automatically within the model before fitting for improved surrogate modelling. 
* Further processing was only performed on the Function 1 observation outputs by projecting them into log space to handle the large range variation and flooring negative values to -130 in the projected space. 

No other data cleaning or pre-processing steps were performed.

### Recommended Uses
* Surrogate modelling
* Bayesian optimisation and gaussian process method development
* Acquisition function research and development

### Out-of-Scope & Limitations
* Neural network training especially deep learning 
* Classification model development

---

## 5. Distribution & Maintenance

### Availability
The dataset is stored on the BBO capstone GitHub repository. New query points are appended at the end of each optimisation round.

### Terms of use
The original data and subsequent query additions have been generated for academic purposes and should only be used within the context of the BBO project.

### Maintenance
The dataset is maintained and updated by the current author. Weekly updates are performed during the duration of the BBO project (February 2026 – September 2026), upon which the dataset will become static.
