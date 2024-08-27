# Introduction

This site contains supporting information for a paper in progress on predicting f-divergence measures from catchment attributes.  Catchment attributes are widely used in hydrological modelling and other environmental applications because of their associations with the hydrological response of catchments.  The f-divergence measures are a family of statistical measures that quantify the difference between two probability distributions.  

Hydrological signatures are minimal representations of process information contained in streamflow observations {cite}`gupta2008reconciling`. There is a large body of literature defining and describing the use of hydrological signatures in various fields of hydrology, see "*A review of hydrological signatures and their applications*" {cite}`mcmillan2021review`. According to  f-divergence measures have not been described in the context of hydrological signatures.  The aim of this work is to investigate the predictability of various f-divergence measures from catchment attributes and provide context for interpreting the impacts of key assumptions.

## Computational Notes

The machine learning technique used to predict the various f-divergence measures from catchment attributes is gradient boosting, and the widely used XGBoost library is used for its implementation{cite}`chen2016xgboost`.  (notes about xgboost).  The XGBoost library facilitates parallel CPU and GPU training, making it feasible to run a large number of models to test sensitivity to key assumptions.  For the data structures tested, the GPU processing doesn't provide notable efficiency gains over CPU.  The first two models predicting single-site target variables (mean annual runoff and entropy) from attributes takes in the order of hours to process, while the much larger datasets used to train the pairwise models, and in particular the KL divergence model, take roughly a day to process on a 14-core Xeon E5-2690 v4 @ 2.60GHz CPU with 128GB DDR4 memory (@2133 MT/s).


## Contents of this book

```{tableofcontents}
```

## Citations 

```{bibliography}
:filter: docname in docnames
```
