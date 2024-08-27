# Introduction

This site contains supporting information for a paper in progress on predicting f-divergence measures from catchment attributes.  Catchment attributes are widely used in hydrological modelling and other environmental applications because of their associations with the hydrological response of catchments.  The f-divergence measures are a family of statistical measures that quantify the difference between two probability distributions.  

Hydrological signatures are minimal representations of process information contained in streamflow observations {cite}`gupta2008reconciling`. There is a large body of literature defining and describing the use of hydrological signatures in various fields of hydrology, see "*A review of hydrological signatures and their applications*" {cite}`mcmillan2021review`. According to  f-divergence measures have not been described in the context of hydrological signatures.  The aim of this work is to investigate the predictability of various f-divergence measures from catchment attributes and provide context for interpreting the impacts of key assumptions.

## Computational Notes

The machine learning technique used to predict the various f-divergence measures from catchment attributes is gradient boosting, and the widely used XGBoost library is used for its implementation{cite}`chen2016xgboost`.  (notes about xgboost).  The XGBoost library facilitates parallel CPU and GPU training, making it feasible to run a large number of models to test sensitivity to key assumptions.  For the data structures tested, training on GPU doesn't provide notable efficiency gains over CPU.  The first two models predicting single-site target variables (mean annual runoff and entropy) from attributes takes roughly an hour to process, while the much larger catchment pair datasets used to train the f-divergence prediction models take longer, with the KL divergence set taking about 15 hours due to the combination of bitrates and priors tested. These processing times correspond to an Intel Xeon E5-2690 v4 @ 2.60GHz CPU.

The catchment attribute validation / reprocessing is a memory intensive step in the data preparation.  Processing the various raster files for the largest catchments is the critical step in memory consumption, and this dataset was processed on a machine with 128GB DDR4 RAM.  The computation-intensive pre-processing steps can be bypassed by downloading the pre-processed data files from the repository linked at the beginning of the Data section.


## Contents of this book

```{tableofcontents}
```

## Citations 

```{bibliography}
:filter: docname in docnames
```
