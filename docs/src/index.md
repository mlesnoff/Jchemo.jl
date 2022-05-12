```@meta
DocTestSetup  = quote
    using Jchemo
end
```

# Jchemo.jl

Documentation for [Jchemo.jl](https://github.com/mlesnoff/Jchemo.jl).

## Overview

**Jchemo** provides elementary functions and pipelines for predictions in chemometrics or other domains. It mainly focuses on methods for high dimensional data. 

Generic functions such as **transform**, **predict**, **coef** and **summary** are available. Tuning the models is facilitated by functions **gridscore** (validation dataset) and **gridcv** (cross-validation), in addition to faster versions for models based on latent variables (LVs) (**gridscorelv** and **gridcvlv**) and ridge regularization (**gridscorelb** and **gridcvlb**).

The package is under construction. Functions may change in the future.

```@autodocs
Modules = [Jchemo]
Order   = [:function, :type]
```

## Domains

### --------- **PCA**
- **pcaeigen** Eigen decomposition
- **pcaeigenk** Eigen decomposition for wide matrices (kernel form)
- **pcasvd** SVD decomposition
- **kpca** Non linear kernel  (KPCA) *Scholkopf et al. 2002*

*Utility (works also for PLS)* 
- **scordis** Score distances (SDs) for a score space
- **odis** Orthogonal distances (ODs) for a score space
- **xfit** Matrix fitting 
- **xresid** Residual matrix 

*Multiblock*
- **blockscal[_col, _frob, _mfa, _sd]** Scaling blocks
- **mbpca_cons** Consensus (CPCA, MBPCA)
- **mbpca_comdim _s** Common components and specific weights (CCSWA, ComDim)
- **rv** RV correlation coefficient
- **lg** Lg coefficient

### --------- **RANDOM PROJECTIONS**

- **rp** Random projection
- **rpmat_gauss** Gaussian random projection matrix 
- **rpmat_li** Sparse random projection matrix 

### --------- **REGRESSION**

#### **Linear models**

*Anova*

- **aov1** One factor ANOVA

*Multiple linear regression (MLR)*

- **mlr** QR algorithm
- **mlrchol** Normal equations and Choleski factorization
- **mlrpinv** Pseudo-inverse
- **mlrpinv_n** Normal equations and pseudo-inverse
- **mlrvec** Simple linear regression (Univariate x)

*Ill-conditionned* 

- **cglsr** Conjugate gradient for the least squares normal equations (CGLS)

#### **Partial least squares (PLSR)**

- **plskern** "Improved kernel #1" *Dayal & McGregor 1997*
- **plsnipals** NIPALS
- **plsrosa** ROSA *Liland et al. 2016*
- **plssimp** SIMPLS *de Jong 1993*
- **plsr_avg** Averaging and stacking PLSR models with different numbers of LVs (PLSR-AVG)
- **kplsr** Non linear kernel (KPLSR) *Rosipal & Trejo 2001*
- **dkplsr** Direct non linear kernel (DKPLSR) *Bennett & Embrechts 2003*

*Utility*
- **aicplsr** AIC and Cp for PLSR

*Variable selection*
- **covsel** CovSel (Roger et al. 2011)
- **iplsr** Interval PLSR (iPLS) (Nørgaard et al. 2000)

*Multiblock*
- **mbplsr** Multiblock (MBPLSR; concatenated autoscaled blocks)
- **mbplsr_rosa** ROSA *Liland et al. 2016*
- **mbplsr_so** Sequentially orthogonalized (SO-PLSR) 

#### **Principal component (PCR)**

- **pcr** SVD factorization

#### **Ridge (RR, KRR)**

- **rr** Pseudo-inverse (RR)
- **rrchol** Choleski factorization (RR)
- **krr** Non linear kernel (KRR) = Least squares SVM (LS-SVMR)

#### **Local models**

- **knnr** kNNR
- **lwplsr** kNN Locally weighted PLSR (kNN-LWPLSR)
- **lwplsr_s** kNN-LWPLSR with preliminary dimension reduction
- **lwplsr_avg** kNN-LWPLSR-AVG 
- **cplsr_avg** Clustered PLSR-AVG

#### **Support vector machine (SVMR)** -- from LIBSVM.jl
- **svmr** Epsilon-SVM regression

#### **Trees** -- from XGBoost.jl

- **treer_xgb** Single tree
- **rfr_xgb** Random forest
- **xgboostr** XGBoost
- **vimp_xgb** Variable importance (Works also for DA models)

#### **Bagging**

- **baggr** Bagging 
- **baggr_oob** Out-of-bag error rate
- **baggr_vi** Variance importance (permutation method)

### --------- **DISCRIMINATION ANALYSIS (DA)**

#### Factorial discrimination analysis (FDA)

- **fda** Eigen decomposition of the compromise "inter/intra"
- **fdasvd** Weighted SVD decomposition of the class centers

#### DA based on predicted Y-dummy table

- **mlrda** On MLR predictions (MLR-DA)
- **plsrda** On PLSR predictions (PLSR-DA; = common "PLSDA")
- **plsrda_avg** Averaging PLSR-DA models with different numbers of LVs (PLSR-DA-AVG)
- **kplsrda** On KPLSR predictions (KPLSR-DA)
- **rrda** On RR predictions (RR-DA)
- **krrda** On KRR predictions (KRR-DA)

#### Probabilistic

- **lda** Linear discriminant analysis (LDA)
- **qda** Quadratic discriminant analysis (QDA)
- **plslda** LDA on PLS latent variables (PLS-LDA)
- **plslda_avg** Averaging PLS-LDA models with different numbers of LVs (PLS-LDA-AVG)
- **plsqda** QDA on PLS latent variables (PLS-QDA)
- **plsqda_avg** Averaging PLS-QDA models with different numbers of LVs (PLS-QDA-AVG)

*Utility*

- **dmnorm** Normal probability density of multivariate data

#### **Local models**

- **knnda** kNN-DA (Vote within neighbors)
- **lwplsrda** kNN Locally weighted PLSR-DA (kNN-LWPLSR-DA)
- **lwplslda** kNN Locally weighted PLS-LDA (kNN-LWPLS-LDA)
- **lwplsqda** kNN Locally weighted PLS-QDA (kNN-LWPLS-QDA)

*Averaging models with different numbers of LVs*

- **lwplsrda_avg** kNN-LWPLSR-DA-AVG
- **lwplslda_avg** kNN-LWPLS-LDA-AVG
- **lwplsqda_avg** kNN-LWPLS-QDA-AVG

#### **Support vector machine (SVM-DA)** -- from LIBSVM.jl
- **svmda** C-SVM discrimination

#### **Trees** -- from XGBoost.jl

- **treeda_xgb** Single tree
- **rfda_xgb** Random forest
- **xgboostda** XGBoost

### --------- **TUNING MODELS**

#### **Grid**

- **mpar** Expand a grid of parameter values

#### **Validation**

- **gridscore** Any model
- **gridscorelv** Models with LVs (faster)
- **gridscorelb** Models with ridge parameter (faster)
  
#### **Cross-validation (CV)**

- **gridcv** Any model
- **gridcvlv** Models with LVs (faster)
- **gridcvlb** Models with ridge parameter (faster)  
- **gridcv_mb** Multiblock models 
- **gridcvlv_mb** Multiblock models with LVs 
- **segmkf** Building segments for K-fold CV
- **segmts** Building segments for test-set validation

#### **Performance scores**

*Regression*

- **ssr** SSR
- **msep** MSEP
- **rmsep** RMSEP
- **sep** SEP
- **bias** Bias
- **cor2** Squared correlation coefficient
- **r2** R2
- **rpd**, **rpdr** Ratio of performance to deviation
- **mse** Summary for regression

*Discrimination*

- **err** Classification error rate

### --------- **DATA MANAGEMENT**

#### **Checking**

- **checkdupl** Finding replicated rows in a dataset
- **checkmiss** Finding rows with missing data in a dataset

#### **Calibration transfert**

- **caltransf_ds** : Direct standardization (DS).
- **caltransf_pds** : Piecewise direct standardization (PDS).

#### **Pre-processing**

- **detrend** Polynomial detrend
- **eposvd** External parameter orthogonalization (EPO)
- **fdif** Finite differences
- **interpl** Sampling signals by intrerpolation -- From DataInterpolations.jl
- **interpl_mon** Sampling signals by monotonic intrerpolation -- From Interpolations.jl
- **mavg**, **mavg_runmean** Smoothing by moving average
- **rmgap** Remove vertical gaps in spectra, e.g. for ASD NIR data
- **savgk**, **savgol** Savitsky-Golay filtering
- **snv** Standard-normal-deviation transformation

#### **Sampling observations**

- **sampdp** Duplex sampling 
- **sampks** Kennard-Stone sampling 
- **sampsys** Systematic sampling
- **sampclas** Stratified sampling

### --------- **PLOTTING**

- **plotsp** Plotting spectra

### --------- **UTILITIES**

- **aggstat** Compute column-wise statistics (e.g. mean), by group
- **center**, **scale** Column-wise centering and scaling of a matrix
- **colmean**, **colnorm2**, **colstd**, **colsum**, **colvar**  Column-wise operations
- **covm**, **corm** Covariance and correlation matrices
- **datasets** Datasets available in the package
- **dummy** Build dummy table
- **euclsq**, **mahsq**, **mahsqchol** Distances (Euclidean, Mahalanobis) between rows of matrices
- **fweight** Compute weights from distances
- **getknn** Find nearest neighbours between rows of matrices
- **iqr** Interval inter-quartiles
- **krbf, kpol** Build kernel Gram matrices
- **locw** Working function for local (kNN) models
- **mad** Median absolute deviation
- **matB**, **matW** Between- and within-covariance matrices
- **mblock** Make blocks from a matrix
- **mweight** Normalize a vector to sum to 1.
- **nco**, **nro**, Nb. rows and colmuns of an object.
- **norm2** Squared norm of a vector
- **recodcat2int** Recode a categorical variable to a numeric variable
- **recodnum2cla** Recode a continuous variable to classes
- **replacebylev** Replace the elements of a vector by levels of corresponding order
- **replacebylev2** : Replace the elements of an index-vector by levels
- **rowmean**, **rowstd**, **rowsum** Row-wise operations
- **sourcedir** Include all the files contained in a directory
- **ssq** Total inertia of a matrix
- **summ** Summarize the columns of a dataset
- **tab**, **tabn** Univariate tabulation 
- **wdist** Compute weights from distances
- Other functions in file `utility.jl`






