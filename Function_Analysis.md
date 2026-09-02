# Optimisation Analysis

## Overall trends

Key metrics regarding the results of the optimisation campaign across all functions are summarised in the table below.

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

The overall optimisation progression based on the submitted queries for each function is shown in the figure below.

<img width="1375" height="1375" alt="Optimisation_progression_all" src="https://github.com/user-attachments/assets/3457bc8e-7d21-4e00-8b1d-4a329071ba6e" />

Considering the results from all functions the following have been observed:
- Round 3 is the worst observation of the entire run for F1, F3, F4, F6, F7 and F8. This is a consequence of a model change introducing ensembling of sci-kit GP fits of a resampled set. The re-sampling was inappropriate, given set sparsity, creating degenerate fits and massively inflating uncertainty. This methodology was removed after this round.
- The implementation library change from sci-kit learn to BoTorch in round 4 is validated by the data with predictions and observations showing considerable improvement and steady progress from round 4 onwards across all functions. 
- The SAAS BoTorch prior applied to the high dimensional functions F7, F8 produced the two highest hit rates (percentage of queries where incumbent improved) showing that the selection was appropriate leading to high function outputs consistently.
- The model performed worst at function 1 failing to capture the underlying structure and best at function 7 showing continuous sizable gains and a consistently increasing incumbent throughout the optimisation campaign.

**All functions managed to improve upon the initial observation best.**

---

## Function Analysis

The following section includes additional plots showing the optimisation progression in more detail for each individual function. Key aspects of the strategy and general observations are discussed

Each function is analysed based on 4 plots:

Plot A: Captures the optimisation progression showing GP predicted mean against observations at each query round

Plot B: A PCA projection of all observations and best point on the two principal component axes. The projection background is contoured by the observation function output values. Initial observations are represented by black x markers. Query points are represented by grayscale circles with the colouring showing the query round that observation was produced. Having this view allows for visualisation of points distribution/clustering along with the optimisation history on the highest variance dimensions.

Plot C: Captures the changes in kernel lengthscale over the course of the optimisation along with the variability in that estimate. The lengthscales in rounds 1-3 come from the sci-kit learn model hence no variability band is available. From round 4 onwards the variability band is present as predicted from the posterior sampling of the BoTorch implementation.

Plot D: On the left axis the metric plotted is a standard deviation fraction calculated via the standard deviation at the query point over the maximum standard deviation in the posterior during that round. This normalisation is used to infer exploration vs exploitation. If close to 1 it means the GP is exploring, if close to 0 it means the GP is exploiting. On this trace points that improved the incumbent are highlighted with green circles. On the right hand axis the model cross-validation NLPD metric is plotted as a relative measure of the improvement in the fit by the end of the optimisation compared to the first round.
 


