# Available methods

## MULTIVARIATE EXPLORATORY ANALYSES

*Principal component analysis (PCA)* 
- **pcaeigen** Eigen decomposition
- **pcaeigenk** Eigen decomposition for wide matrices (kernel form)
- **pcasvd** SVD decomposition
- Variants for dimension reduction
    - **rp** Random projection
    - **rpmatgauss** Gaussian random projection matrix 
    - **rpmatli** Sparse random projection matrix 

*Non linear PCA*
- **kpca** Kernel (KPCA) *Scholkopf et al. 2002*

*Multiblock*
- 2 blocks
    - **cca**: Canonical correlation analysis (CCA)
    - **ccawold**: CCA - Wold (1984) Nipals algorithm  
    - **plscan**: Canonical partial least squares regression (Symmetric PLS)
    - **plstuck**: Tucker's inter-battery method of factor analysis (PLS-SVD)
    - **rasvd**: Redundancy analysis - PCA on instrumental variables (PCAIV)
- 2 or more blocks 
    - **mbpca** Consensus PCA (MBPCA = CPCA)
    - **comdim** Common components and specific weights analysis (ComDim = CCSWA = HPCA)
    - **mbmang**: Unified multiblock data analysis of Mangana et al. 2019
- Utilities
    - **mblock** Make blocks from a matrix
    - **blockscal_col, _frob, _mfa, _sd** Scaling blocks
    - **rd** Redundancy coefficients between two matrices
    - **lg** Lg coefficient
    - **rv** RV correlation coefficient

*Utilities for PCA and PLS* 
- **xfit** Matrix fitting 
- **xresid** Residual matrix 

## REGRESSION

### **Linear models**

*Multiple linear regression (MLR)*
- **mlr** QR algorithm
- **mlrchol** Normal equations and Choleski factorization
- **mlrpinv** Pseudo-inverse
- **mlrpinvn** Normal equations and pseudo-inverse
- **mlrvec** Simple linear regression (Univariate x)

*Anova*
- **aov1** One factor ANOVA

### **Partial least squares (PLSR)**

*PLSR*
- **plskern** "Improved kernel #1" *Dayal & McGregor 1997* (Fast)
- **plsnipals** Nipals
- **plswold** Nipals *Wold 1984*
- **plsrosa** ROSA *Liland et al. 2016*
- **plssimp** SIMPLS *de Jong 1993*

*Variants* 
- **cglsr** Conjugate gradient for the least squares normal equations (CGLS)
- **ramang** Redundancy analysis regression = Reduced rank regression (RRR)
- **pcr** Principal components regression (SVD factorization)
- **covselr** MLR on variables selected from Covsel

*Non linear*
- **kplsr** Non linear kernel (KPLSR) *Rosipal & Trejo 2001*
- **dkplsr** Direct non linear kernel (DKPLSR) *Bennett & Embrechts 2003*

*Averaging and stacking*
- **plsravg** Averaging and stacking PLSR models with different numbers of 
    latent variables (LVs) (PLSR-AVG)

*Variable selection*
- **iplsr** Interval PLSR (iPLS) (Nørgaard et al. 2000)
- **covsel** Variable selection from partial correlation or covariance (Covsel)

*Multiblock*
- **mbplsr** Multiblock (MBPLSR; concatenated blocks)
- **rosaplsr** ROSA *Liland et al. 2016*
- **soplsr** Sequentially orthogonalized (SO-PLSR) 

### **Ridge (RR, KRR)**

*RR*
- **rr** Pseudo-inverse (RR)
- **rrchol** Choleski factorization (RR)

*Non linear*
- **krr** Non linear kernel (KRR) = Least squares SVM (LS-SVMR)

### **Local models**

- **knnr** kNNR
- **lwplsr** kNN Locally weighted PLSR (kNN-LWPLSR)
- **lwplsr_s** kNN-LWPLSR with preliminary dimension reduction

*Averaging and stacking*
- **lwplsravg** kNN-LWPLSR-AVG 
- **cplsravg** Clustered PLSR-AVG

### **Support vector machine (SVMR)** -- from LIBSVM.jl

- **svmr** Epsilon-SVM regression

### **Trees** -- from XGBoost.jl

- **treer_xgb** Single tree
- **rfr_xgb** Random forest
- **xgboostr** XGBoost
- **vimp_xgb** Variable importance (Works also for DA models)

### **Generic bagging**

- **baggr** Bagging 
- **baggr_oob** Out-of-bag error rate
- **baggr_vi** Variance importance (permutation method)

## DISCRIMINATION ANALYSIS (DA)

### One-Class Classification (OCC)

*From a PCA or PLS score space*
- **occsd** Using score distance (SD)
- **occod** Using orthogonal distance (OD) 
- **occsdod** Using a compromise between SD and OD 

*Other methods*
- **occknndis**: Using "global" k-nearest neighbors distances.
- **occlknndis**: Using "local" k-nearest neighbors distances.
- **occstah** Using Stahel-Donoho outlierness measure.
- **stah** Compute Stahel-Donoho outlierness measure.

### Factorial discrimination analysis (FDA)

- **fda** Eigen decomposition of the compromise "inter/intra"
- **fdasvd** Weighted SVD decomposition of the class centers

### DA based on predicted Y-dummy table

- **mlrda** On MLR predictions (MLR-DA)
- **plsrda** On PLSR predictions (PLSR-DA; = common "PLSDA")
- **plsrdaavg** Averaging PLSR-DA models with different numbers of 
    latent variables (LVs) (PLSR-DA-AVG)
- **rrda** On RR predictions (RR-DA)

*Non linear*
- **kplsrda** On KPLSR predictions (KPLSR-DA)
- **krrda** On KRR predictions (KRR-DA)

### Probabilistic

- **lda** Linear discriminant analysis (LDA)
- **qda** Quadratic discriminant analysis (QDA)
- **plslda** LDA on PLS latent variables (PLS-LDA)
- **plsqda** QDA on PLS latent variables (PLS-QDA)
- **plsqdaavg** Averaging PLS-QDA models with different numbers of 
    latent variables (LVs) (PLS-QDA-AVG)

*Averaging*
- **plsldaavg** Averaging PLS-LDA models with different numbers of 
    latent variables (LVs) (PLS-LDA-AVG)

*Utility*
- **dmnorm** Normal probability density of multivariate data

### **Local models**

- **knnda** kNN-DA (Vote within neighbors)
- **lwplsrda** kNN Locally weighted PLSR-DA (kNN-LWPLSR-DA)
- **lwplslda** kNN Locally weighted PLS-LDA (kNN-LWPLS-LDA)
- **lwplsqda** kNN Locally weighted PLS-QDA (kNN-LWPLS-QDA)

*Averaging*
- **lwplsrdaavg** kNN-LWPLSR-DA-AVG
- **lwplsldaavg** kNN-LWPLS-LDA-AVG
- **lwplsqdaavg** kNN-LWPLS-QDA-AVG

### **Support vector machine (SVM-DA)** -- from LIBSVM.jl
- **svmda** C-SVM discrimination

### **Trees** -- from XGBoost.jl

- **treeda_xgb** Single tree
- **rfda_xgb** Random forest
- **xgboostda** XGBoost

## TUNING MODELS

### **Grid**

- **mpar** Expand a grid of parameter values

### **Validation**

- **gridscore** Any model
- **gridscorelv** Models with LVs (faster)
- **gridscorelb** Models with ridge parameter (faster)
  
### **Cross-validation (CV)**

- **gridcv** Any model
- **gridcvlv** Models with LVs (faster)
- **gridcvlb** Models with ridge parameter (faster)  
- **gridcv_mb** Multiblock models 
- **gridcvlv_mb** Multiblock models with LVs 
- **segmkf** Building segments for K-fold CV
- **segmts** Building segments for test-set validation

### **Performance scores**

*Regression*
- **ssr** SSR
- **msep** MSEP
- **rmsep**, **rmsepstand** RMSEP
- **sep** SEP
- **bias** Bias
- **cor2** Squared correlation coefficient
- **r2** R2
- **rpd**, **rpdr** Ratio of performance to deviation
- **mse** Summary for regression

*Discrimination*
- **err** Classification error rate

*Model dimensionality*
- **aicplsr** AIC and Cp for PLSR
- **selwold** Wold's criterion to select dimensionality in LV (e.g. PLSR) models

## DATA MANAGEMENT

### **Checking**

- **checkdupl** Finding replicated rows in a dataset
- **checkmiss** Finding rows with missing data in a dataset

### **Calibration transfert**

- **calds** : Direct standardization (DS).
- **calpds** : Piecewise direct standardization (PDS).

### **Pre-processing**

- **detrend** Polynomial detrend
- **eposvd** External parameter orthogonalization (EPO)
- **fdif** Finite differences
- **mavg**, **mavg_runmean** Smoothing by moving average
- **rmgap** Remove vertical gaps in spectra, e.g. for ASD NIR data
- **savgk**, **savgol** Savitsky-Golay filtering
- **snv** Standard-normal-deviation transformation

*Interpolation*
- **interpl** Sampling signals by interpolation -- From DataInterpolations.jl
- **interpl_mon** Sampling signals by monotonic interpolation -- From Interpolations.jl

### **Sampling observations**

- **sampdp** Duplex sampling 
- **sampks** Kennard-Stone sampling 
- **sampsys** Systematic sampling
- **sampclas** Stratified sampling

## PLOTTING

- **plotgrid** Ploting error or performance rates of model predictions
- **plotsp** Ploting spectra
- **plotxy** Scatter plot of (x, y) data

## UTILITIES

- **aggstat** Compute column-wise statistics (e.g. mean), by group
- **center**, **scale**, **cscale** Column-wise centering and scaling of a matrix
- **colmad**, **colmean**, **colnorm**, **colstd**, **colsum**, **colvar**  Column-wise operations
- **covm**, **corm** Covariance and correlation matrices
- **datasets** Datasets available in the package
- **dens** Univariate kernel density estimation
- **dummy** Build dummy table
- **euclsq**, **mahsq**, **mahsqchol** Distances (Euclidean, Mahalanobis) between rows of matrices
- **frob** Frobenius norm of a matrix
- **fweight** Compute weights from distances
- **getknn** Find nearest neighbours between rows of matrices
- **iqr** Interval inter-quartiles
- **krbf, kpol** Build kernel Gram matrices
- **locw** Working function for local (kNN) models
- **mad** Median absolute deviation
- **matB**, **matW** Between- and within-covariance matrices
- **mweight** Normalize a vector to sum to 1.
- **nco**, **nro**, Nb. rows and colmuns of an object.
- **normw** Weighted norm of a vector
- **recodcat2int** Recode a categorical variable to a numeric variable
- **recodnum2cla** Recode a continuous variable to classes
- **replacebylev** Replace the elements of a vector by levels of corresponding order
- **replacebylev2** : Replace the elements of an index-vector by levels
- **replacedict** : Replace the elements of a vector by levels defined in a dictionary
- **rowmean**, **rowstd**, **rowsum** Row-wise operations
- **sourcedir** Include all the files contained in a directory
- **ssq** Total inertia of a matrix
- **summ** Summarize the columns of a dataset
- **tab**, **tabn** Univariate tabulation 
- **wdist** Compute weights from distances
- Other functions in file `utility.jl`


