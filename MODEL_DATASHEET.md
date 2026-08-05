# Model Datasheet

## 1. Motivation

### Why was the dataset created?
The dataset was created to capture the observations used in the optimisation of 8 unknown black box functions as part of a black box optimisation (BBO) project. The project objective was to maximise each individual function under a limited query budget of 12 evaluations.

### What task does it support? 
The dataset supports Bayesian optimisation, surrogate modelling using gaussian process regressors, acquisition function tuning and optimisation and next candidate selection strategies for optimisation of black box functions.

---

## 2. Dataset Composition

The dataset consists of the initial observations prior to the start of the optimisation and the round by round observations after each new function evaluation was performed.

### Contents

* **Total Size / Scale:**
The size of the original observation dataset varies between 10 – 40 points depending on the function. Overall dataset size after 12 evaluations increases to 22 – 62 observations. Black box function input dimensionality varies from 2 to 8.
* **Data Types:**
Both inputs and outputs are continuous numerical values. Inputs range within the unit cube [0,1]^d across all functions whereas output range can be very different and spans multiple orders of magnitude.
* **File Formats:**
All data points are stored in NumPy arrays and text files

### Observations
An individual observation consists of a pair of input and output variables. The inputs are represented by a vector with length equal to the function dimensions and the output is a single value representing the black box function output value at the specified input.

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

The timeframe of the project was 12 weeks. One evaluation per week for each function was permitted

### Strategy

The optimisation strategy evolved within the duration of the project.

Early rounds focused on exploration of the design space targeting reduction of overall uncertainty in the model fit. Later rounds transitioned into exploitation of the promising regions that were being progressively identified. 

The transition point was different for each function depending on function dimensionality, the outputs of each round and model fit metrics. 

---

## 4. Preprocessing & Uses

### Pre-processing

Input variable scaling was applied where relevant. No other data cleaning or pre-processing steps were performed.

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
