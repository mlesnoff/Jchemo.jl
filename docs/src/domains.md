# Available methods

## MULTIVARIATE EXPLORATORY DATA ANALYSES

### Principal component analysis (PCA) 

*Usual*
- **pcaeigen** Eigen decomposition
- **pcaeigenk** Eigen decomposition for wide matrices (kernel form)
- **pcanipals** NIPALS algorithm
- **pcasvd** SVD decomposition

*With missing data*
- **pcanipalsmiss**: NIPALS algorithm allowing missing data

*Robust* 
- **pcasph** Spherical (with spatial median)

*Sparse* 
- **spca** sPCA *Shen & Huang 2008*

*Non linear*

- **kpca** Kernel (KPCA) *Scholkopf et al. 2002*

*Utilities (PCA and PLS)* 
- **xfit** Matrix fitting 
- **xresid** Residual matrix 

### Random projections

- **rp** Random projection
- **rpmatgauss** Gaussian random projection matrix 
- **rpmatli** Sparse random projection matrix 

### Multiblock

*2 blocks*
- **cca** Canonical correlation analysis (CCA, RCCA)
- **ccawold** CCA, RCCA - Wold (1984) Nipals algorithm  
- **plscan** Canonical partial least squares regression (Symmetric PLS)
- **plstuck** Tucker's inter-battery method of factor analysis (PLS-SVD)
- **rasvd** Redundancy analysis (RA) - PCA on instrumental variables (PCAIV)

*2 or more blocks* 
- **mbpca** Multiblock PCA (MBPCA = CPCA Consensus principal component analysis)
- **comdim** Common components and specific weights analysis (ComDim = CCSWA = HPCA)
- **mbunif** Unified multiblock data analysis of Mangana et al. 2019

*Utilities*
- **mblock** Make blocks from a matrix
- **blockscal_col, _frob, _mfa, _sd** Scaling blocks
- **rd** Redundancy coefficients between two matrices
- **lg** Lg coefficient
- **rv** RV correlation coefficient

### Factorial discrimination analysis (FDA)

- **fda** Eigen decomposition of the compromise "inter/intra"
- **fdasvd** Weighted SVD decomposition of the class centers

## REGRESSION

### **Ordinary least squares (OLS)**

*Multiple linear regression (MLR)*
- **mlr** QR algorithm
- **mlrchol** Normal equations and Choleski factorization
- **mlrpinv** Pseudo-inverse
- **mlrpinvn** Normal equations and pseudo-inverse
- **mlrvec** Simple linear regression (Univariate x)

*Anova*
- **aov1** One factor ANOVA

### **Partial least squares (PLSR)**

*Usual (asymetric regression mode)*
- **plskern** Fast "improved kernel #1" algorithm of *Dayal & McGregor 1997*
- **plsnipals** Nipals
- **plswold** Nipals *Wold 1984*
- **plsrosa** ROSA *Liland et al. 2016*
- **plssimp** SIMPLS *de Jong 1993*

*Variants of regularization using latent variables* 
- **cglsr** Conjugate gradient for the least squares normal equations (CGLS)
- **rrr** Reduced rank regression (RRR)  (= redundancy analysis regression) 
- **pcr** Principal components regression (SVD factorization)

*Sparse*
- **splskern** sPLSR *Lê Cao et al. 2008*
- **covselr** MLR on variables selected from Covsel method *Roger et al. 2011*

*Averaging and stacking of PLSR models of different dimensionalities*
- **plsravg** PLSR-AVG

*Non linear*
- **kplsr** Non linear kernel (KPLSR) *Rosipal & Trejo 2001*
- **dkplsr** Direct non linear kernel (DKPLSR) *Bennett & Embrechts 2003*

*Multiblock*
- **mbplsr** Multiblock PLSR (MBPLSR) - Fast version (PLSR on concatenated blocks)
- **mbplswest** MBPLSR - Nipals algorithm *Westerhuis et al. 1998* 
- **mbwcov** Multiblock weighted covariate analysis regression (MBWCov) *Mangana et al. 2021* 
- **rosaplsr** ROSA *Liland et al. 2016*
- **soplsr** Sequentially orthogonalized (SO-PLSR) 

### **Ridge (RR, KRR)**

*RR*
- **rr** Pseudo-inverse (RR)
- **rrchol** Choleski factorization (RR)

*Non linear*
- **krr** Non linear kernel (KRR) = Least squares SVM (LS-SVMR)

### **Local models**

- **knnr** kNN regression
- **lwmlr** kNN locally weighted MLR (kNN-LWMLR)
- **lwplsr** kNN locally weighted PLSR (kNN-LWPLSR)

*With preliminary dimension reduction*
- **lwmlr_s** kNN-LWMLR-S
- **lwplsr_s** kNN-LWPLSR-S

*Averaging and stacking*
- **lwplsravg** kNN-LWPLSR-AVG 
- **cplsravg** Clustered PLSR-AVG

### **Generic bagging**

- **baggr** Bagging 
- **oob_baggr** Out-of-bag (OOB) error rate

### Wrappers to other packages

*SVM regression with LIBSVM.jl*

- **svmr** Epsilon-SVR (SVM-R)

*Regression trees with DecisionTree.jl*
- **treer_dt** Single tree
- **rfr_dt** Random forest

## DISCRIMINATION ANALYSIS (DA)

### Based on the prediction of the Y-dummy table

*Linear*
- **mlrda** MLR prediction (MLR-DA)
- **plsrda** PLSR prediction (PLSR-DA) = usual **PLSDA**
- **plsrdaavg** Averaging PLSR-DA models with different numbers of latent variables (PLSR-DA-AVG)
- **rrda** RR prediction (RR-DA)

*Sparse*
- **splsrda** Sparse PLSR-DA

*Non linear*
- **kplsrda** KPLSR prediction (KPLSR-DA)
- **dkplsrda** DKPLSR prediction (DKPLSR-DA)
- **krrda** KRR prediction (KRR-DA)

### Probabilistic DA

*Parametric*
- **lda** Linear discriminant analysis (LDA)
- **qda** Quadratic discriminant analysis (QDA, with continuum towards LDA)
- **rda** Regularized discriminant analysis (RDA)
- **nscda**: DA by nearest shrunken centroids (Tibshirani et al. 2002, 2003)

*Non parametric*
- **kdeda** DA by kernel Gaussian density estimation (KDE-DA)

*On PLS latent variables*
- **plslda** PLS-LDA
- **plsqda** PLS-QDA (with continuum)
- **plskdeda**  PLS-KDE-DA

*Sparse*
- **splslda**: Sparse PLS-LDA.
- **splsqda**: Sparse PLS-QDA.
- **splskdeda**: Sparse PLS-KDE-DA.

*Averaging PLS-LDA and -QDA models of different dimensionalities*
- **plsldaavg** PLS-LDA-AVG
- **plsqdaavg** PLS-QDA-AVG (with continuum)

### **Local models**

- **knnda** kNN-DA (Vote within neighbors)
- **lwmlrda** kNN locally weighted MLR-DA (kNN-LWMLR-DA)
- **lwplsrda** kNN Locally weighted PLSR-DA (kNN-LWPLSR-DA)
- **lwplslda** kNN Locally weighted PLS-LDA (kNN-LWPLS-LDA)
- **lwplsqda** kNN Locally weighted PLS-QDA (kNN-LWPLS-QDA, with continuum)

*With preliminary dimension reduction*
- **lwmlrda_s** kNN-LWMLR-DA-S
- **lwplsrda_s** kNN-LWPLSR-DA-S

*Averaging*
- **lwplsrdaavg** kNN-LWPLSR-DA-AVG
- **lwplsldaavg** kNN-LWPLS-LDA-AVG
- **lwplsqdaavg** kNN-LWPLS-QDA-AVG (with continuum)

### One-Class Classification (OCC)

*From a PCA or PLS score space*
- **occsd** Score distance (SD)
- **occod** Orthogonal distance (OD) 
- **occsdod** Compromise between SD and OD (Simca approach) 

*Other methods*
- **stah** Compute Stahel-Donoho outlierness
- **occstah** Stahel-Donoho outlierness
- **occknndis** Global k-nearest neighbors distances
- **occlknndis** Local k-nearest neighbors distances

### Wrappers to other packages

*SVM classification with LIBSVM.jl*

- **svmda** C-SVC (SVM-DA)

*Classification trees with DecisionTree.jl*

- **treeda_dt** Single tree
- **rfda_dt** Random forest

## DISTRIBUTIONS

- **dmnorm** Normal probability density estimation
- **dmnormlog** Logarithm of the normal probability density estimation
- **dmkern** Gaussian kernel density estimation (KDE)
- **pval** Compute p-value(s) for a distribution, a vector or an ECDF
- **out** Return if elements of a vector are strictly outside of a given range

## VARIABLE IMPORTANCE

- **isel** Interval variable selection (e.g. Interval PLSR).
- **nsc**: Nearest shrunken centroids (and variable selection).
- **vip** Variable importance on projections (VIP)
- **vi_baggr** Variable importance after bagging (OOB permutations method)
- **viperm** Variable importance by direct permutations

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
- **confusion** Confusion matrix

*Discrimination*
- **err** Classification error rate

*Model dimensionality*
- **aicplsr** AIC and Cp for PLSR
- **selwold** Wold's criterion to select dimensionality in LV (e.g. PLSR) models

## DATA MANAGEMENT

### **Checking**

- **dupl** Finding replicated rows in a dataset
- **tabdupl** Tabulate duplicated values in a vector
- **miss** Finding rows with missing data in a dataset

### **Calibration transfer**

- **calds** Direct standardization (DS)
- **calpds** Piecewise direct standardization (PDS)
- **difmean** Compute a detrimental matrix (for calibration transfer) by 
    difference of two matrix-column means.
- **eposvd** Compute an orthogonalization matrix for calibration transfer

### **Pre-processing**

- **detrend** Polynomial detrend
- **fdif** Finite differences
- **mavg**, **mavg_runmean** Smoothing by moving average
- **rmgap** Remove vertical gaps in spectra, e.g. for ASD NIR data
- **savgk**, **savgol** Savitsky-Golay filtering
- **snv** Standard-normal-deviation transformation

*Interpolation*
- **interpl** Sampling signals by interpolation -- From DataInterpolations.jl
- **interpl_mon** Sampling signals by monotonic interpolation -- From Interpolations.jl

### **Split training/test sets by sampling**

- **samprand** Random (no replacement)
- **sampks** Kennard-Stone 
- **sampdp** Duplex  
- **sampsys** Systematic over a quantitative variable
- **sampcla** Stratified by class
- **mtest** Split for each column of a dataframe (typically, response variables 
    to predict) that can contain missing values

## PLOTTING

- **plotconf** Plot confusion matrix
- **plotgrid** Plot error or performance rates of model predictions
- **plotsp** Plot spectra
- **plotxy** xy scatter plot

## UTILITIES

- **aggstat** Compute column-wise statistics (e.g. mean), by group in a dataset
- **center**, **scale**, **cscale** Column-wise centering and scaling of a matrix
- **colmad**, **colmean**, **colnorm**, **colstd**, **colsum**, **colvar**  Column-wise operations
- **colmeanskip**, **colstdskip**, **colsumskip**, **colvarskip**: Column-wise operations allowing missing data
- **covm**, **corm** Weighted covariance and correlation matrices
- **cosv**, **cosm** Cosinus between vectors
- **dummy** Build dummy table
- **euclsq**, **mahsq**, **mahsqchol** Distances (Euclidean, Mahalanobis) between rows of matrices
- **findmax_cla** Find the most occurent level in a categorical variable
- **frob** Frobenius norm of a matrix
- **fweight** Compute weights from distances
- **getknn** Find nearest neighbours between rows of matrices
- **head**, **@head** Display the first rows of a dataset
- **iqr** Interval inter-quartiles
- **krbf, kpol** Build kernel Gram matrices
- **locw** Working function for local (kNN) models
- **mad** Median absolute deviation (not exported)
- **matB**, **matW** Between- and within-class covariance matrices
- **mlev** Return the sorted levels of a dataset 
- **mweight** Normalize a vector to sum to 1
- **nco**, **nro**, Nb. rows and columns of an object
- **normw** Weighted norm of a vector
- **plist** Print each element of a list
- **pnames** Return the names of the elements of an object
- **psize** Return the type and size of a dataset
- **recodcat2int** Recode a categorical variable to a numeric variable
- **recodnum2cla** Recode a continuous variable to classes
- **replacebylev** Replace the elements of a vector by levels of corresponding order
- **replacebylev2** Replace the elements of an index-vector by levels
- **replacedict** Replace the elements of a vector by levels defined in a dictionary
- **rmcol** Remove the columns of a matrix or the components of a vector having indexes s
- **rmrow** Remove the rows of a matrix or the components of a vector having indexes s
- **rowmean**, **rowstd**, **rowsum**, **rowvar**: Row-wise operations
- **rowmeanskip**, **rowstdskip**, **rowsumskip**, **rowvarskip**: Row-wise operations allowing missing data
- **soft** Soft thresholding
- **softmax** Softmax function
- **sourcedir** Include all the files contained in a directory
- **ssq** Total inertia of a matrix
- **summ** Summarize the columns of a dataset
- **tab**, **tabdf**, **tabdupl** Tabulations for categorical variables
- **vcatdf** Vertical concatenation of a list of dataframes
- **wdist** Compute weights from distances
- Other **utility functions** in file `utility.jl`


