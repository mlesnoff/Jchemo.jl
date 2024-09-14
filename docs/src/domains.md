# Available methods

## MULTIVARIATE EXPLORATORY DATA ANALYSES

### Principal component analysis (PCA) 

*Usual*
- **pcasvd** SVD decomposition
- **pcaeigen** Eigen decomposition
- **pcaeigenk** Eigen decomposition for wide matrices (kernel form)
- **pcanipals** NIPALS algorithm

*Allow missing data*
- **pcanipalsmiss**: NIPALS algorithm allowing missing data

*Robust* 
- **pcasph** Spherical (with spatial median)
- **pcapp** Projection pursuit.
- **pcaout** Outlierness

*Sparse* 
- **spca** sPCA *Shen & Huang 2008*

*Non linear*

- **kpca** Kernel (KPCA) *Scholkopf et al. 2002*

*Utilities (PCA and PLS)* 
- **xfit** X-matrix fitting 
- **xresid** X-residual matrix 

### Random projections

- **rp** Random projection
- **rpmatgauss** Gaussian random projection matrix 
- **rpmatli** Sparse random projection matrix 

### Manifold 

*UMAP* -- with UMAP.jl

- **umap**: Uniform manifold approximation and projection for 
    dimension reduction

### Factorial discrimination analysis (FDA)

- **fda** Eigen decomposition of the compromise "inter/intra"
- **fdasvd** Weighted SVD of the class centers

### Multiblock

*2 blocks*
- **cca** Canonical correlation analysis (CCA and RCCA)
- **ccawold** CCA and RCCA - Wold (1984) Nipals algorithm  
- **plscan** Canonical partial least squares regression (Symmetric PLS)
- **plstuck** Tucker's inter-battery method of factor analysis (PLS-SVD)
- **rasvd** Redundancy analysis (RA), *aka* PCA on 
    instrumental variables (PCAIV)

*2 or more blocks* 
- **mbconcat** Transformer concatenating multi-block X-data
- **mbpca** Multiblock PCA (MBPCA), *aka* Consensus principal component analysis (CPCA)
- **comdim** Common components and specific weights analysis (ComDim), *aka* CCSWA or HPCA

*Utilities*
- **mblock** Make blocks from a matrix
- **rd** Redundancy coefficients between two matrices
- **lg** Lg coefficient
- **rv** RV correlation coefficient

## REGRESSION

### Ordinary least squares (OLS)

*Multiple linear regression (MLR)*
- **mlr** QR algorithm
- **mlrchol** Normal equations and Choleski factorization
- **mlrpinv** Pseudo-inverse
- **mlrpinvn** Normal equations and pseudo-inverse
- **mlrvec** Simple linear regression (Univariate x)

*Anova*
- **aov1** One-factor ANOVA

### Partial least squares (PLSR)

*Usual (asymetric regression mode)*
- **plskern** Fast "improved kernel #1" algorithm of *Dayal & McGregor 1997*
- **plsnipals** Nipals
- **plswold** Nipals *Wold 1984*
- **plsrosa** ROSA *Liland et al. 2016*
- **plssimp** SIMPLS *de Jong 1993*

*Variants of regularization using latent variables* 
- **cglsr** Conjugate gradient for the least squares normal equations (CGLS)
- **pcr** Principal components regression (SVD factorization)
- **rrr** Reduced rank regression (RRR), *aka*  Redundancy analysis regression 

*Robust*
- **plsrout** Outlierness

*Sparse*
- **splskern** 
    - sPLSR *Lê Cao et al. 2008*
    - Covsel regression *Roger et al. 2011*

*Averaging PLSR models of different dimensionalities*
- **plsravg** PLSR-AVG

*Non linear*
- **kplsr** Non linear kernel (KPLSR) 
    *Rosipal & Trejo 2001*
- **dkplsr** Direct non linear kernel (DKPLSR) *Bennett & Embrechts 2003*

*Multiblock*
- **mbplsr** Multiblock PLSR (MBPLSR) - Fast version (PLSR on concatenated blocks)
- **mbplswest** MBPLSR - Nipals algorithm *Westerhuis et al. 1998* 
- **rosaplsr** ROSA *Liland et al. 2016*
- **soplsr** Sequentially orthogonalized (SO-PLSR) 

### Ridge (RR, KRR)

*RR*
- **rr** SVD factorization
- **rrchol** Choleski factorization

*Non linear*
- **krr** Non linear kernel (KRR), *aka* Least squares SVM (LS-SVMR)

### Local models

- **loessr** LOESS regression model -- With package Loess.jl

*kNN*
- **knnr** kNN weighted regression (kNNR)
- **lwmlr** kNN locally weighted MLR (kNN-LWMLR)
- **lwplsr** kNN locally weighted PLSR (kNN-LWPLSR)

*Averaging*
- **lwplsravg** kNN-LWPLSR-AVG 

### Support vector machines -- with LIBSVM.jl

- **svmr** Epsilon-SVR (SVM-R)

### Trees -- with DecisionTree.jl

- **treer** Single tree
- **rfr** Random forest

## DISCRIMINATION ANALYSIS (DA)

### Based on the prediction of the Y-dummy table

*Linear*
- **mlrda** MLR-DA
- **plsrda** PLSR-DA, *aka* usual PLSDA
- **rrda** RR-DA

*Sparse*
- **splsrda** Sparse PLSR-DA

*Non linear*
- **kplsrda** KPLSR-DA
- **dkplsrda** DKPLSR-DA
- **krrda** KRR-DA

*Multiblock* 

- **mbplsrda** MBPLSR-DA

### Probabilistic DA

*Parametric*
- **lda** Linear discriminant analysis (LDA)
- **qda** Quadratic discriminant analysis (QDA, with continuum towards LDA)
- **rda** Regularized discriminant analysis (RDA)

*Non parametric*
- **kdeda** DA by kernel Gaussian density estimation (KDE-DA)

*On PLS latent variables*

* *PLSDA*
    - **plslda** PLS-LDA
    - **plsqda** PLS-QDA (with continuum)
    - **plskdeda**  PLS-KDEDA

* *Sparse*
    - **splslda**: Sparse PLS-LDA
    - **splsqda**: Sparse PLS-QDA
    - **splskdeda**: Sparse PLS-KDEDA

* *Non linear*
    - **kplslda** KPLS-LDA
    - **kplsqda** KPLS-QDA
    - **kplskdeda** KPLS-KDEDA
    - **dkplslda** Direct KPLS-LDA
    - **dkplsqda** Direct KPLS-QDA
    - **dkplskdeda** Direct KPLS-KDEDA

* *Multiblock* 
    - **mbplslda** MBPLS-LDA
    - **mbplsqda** MBPLS-QDA
    - **mbplskdeda** MBPLS-KDEDA

### Local models

- **knnda** kNN-DA (Vote within neighbors)
- **lwmlrda** kNN locally weighted MLR-DA (kNN-LWMLR-DA)
- **lwplsrda** kNN Locally weighted PLSR-DA (kNN-LWPLSR-DA)
- **lwplslda** kNN Locally weighted PLS-LDA (kNN-LWPLS-LDA)
- **lwplsqda** kNN Locally weighted PLS-QDA (kNN-LWPLS-QDA, with continuum)

### Support vector machines -- with LIBSVM.jl

- **svmda** C-SVC (SVM-DA)

### Trees -- with DecisionTree.jl

- **treeda** Single tree
- **rfda** Random forest

## ONE-CLASS CLASSIFICATION (OCC)

### From a PCA or PLS score space

- **occsd** Score distance (SD)
- **occod** Orthogonal distance (OD) 
- **occsdod** Compromise between SD and OD (*aka* Simca approach) 

### Other methods

- **occstah** Stahel-Donoho outlierness

### Utilities

- **outstah** Stahel-Donoho outlierness
- **outeucl**: Outlierness from Euclidean distances to center

## DISTRIBUTIONS

- **dmnorm** Normal probability density estimation
- **dmnormlog** Logarithm of the normal probability density estimation
- **dmkern** Gaussian kernel density estimation (KDE)
- **pval** Compute p-value(s) for a distribution, a vector or an ECDF
- **out** Return if elements of a vector are strictly outside of a given range

## VARIABLE IMPORTANCE

- **isel!** Interval variable selection (e.g. Interval PLSR).
- **vip** Variable importance on projections (VIP)
- **viperm!** Variable importance by direct permutations

## TUNING MODELS

### Test-set validation

- **gridscore** Compute an error rate over a grid of parameters
  
### Cross-validation (CV)

- **gridcv** Compute an error rate over a grid of parameters

### Utilities

- **mpar** Expand a grid of parameter values
- **segmkf** Build segments for K-fold CV
- **segmts** Build segments for test-set validation

### Performance scores

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
- **conf** Confusion matrix

*Discrimination*
- **errp** Classification error rate
- **merrp** Mean intra-class classification error rate

*Model dimensionality*
- **aicplsr** AIC and Cp for PLSR
- **selwold** Wold's criterion to select dimensionality in LV models (e.g. PLSR)

## DATA PROCESSING

### Checking

- **dupl** Find replicated rows in a dataset
- **tabdupl** Tabulate duplicated values in a vector
- **miss** Find rows with missing data in a dataset

### Pre-processing

- De-trend transformation (baseline correction)
    - **dtpol** Polynomial linear regression
    - **dtlo** LOESS
    - **dtasls** Asymmetric least squares (ASLS)
    - **dtairpls** Adaptive iteratively reweighted penalized least squares (AIRPLS)
    - **dtarpls** Asymmetrically reweighted penalized least squares smoothing (ARPLS)
- **snv** Standard-normal-deviation transformation
- **fdif** Finite differences
- **mavg** Smoothing by moving average
- **savgk**, **savgol** Savitsky-Golay filtering
- **rmgap** Remove vertical gaps in spectra, e.g. for ASD NIR data

*Scaling*

- **center** Column centering
- **scale** Column scaling
- **cscale** Column centering and scaling

- **blockscal** Scaling of multiblock data

### Interpolation

- **interpl** Sampling spectra by interpolation 
    -- From DataInterpolations.jl

### Calibration transfer

- **difmean** Compute a detrimental matrix (for calibration transfer) by difference of 
    two matrix-column means.
- **eposvd** Compute an orthogonalization matrix for calibration transfer
- **calds** Direct standardization (DS)
- **calpds** Piecewise direct standardization (PDS)

### Build training vs. test sets by sampling

- **samprand** Random (without replacement)
- **sampsys** Systematic over a quantitative variable
- **sampcla** Stratified by class
- **sampdf** From each column of a dataframe (where missing values are allowed)

- **sampks** Kennard-Stone 
- **sampdp** Duplex  
- **sampwsp** WSP

## PLOTTING

- **plotsp** Plot spectra
- **plotxy** x-y scatter plot
- **plotgrid** Plot error/performance rates of a model
- **plotconf** Plot confusion matrix

## MODELS AND PIPELINES

- **model** Build a model
- **pip** Build a pipeline of models

## UTILITIES

- **aggstat** Compute column-wise statistics by class in a dataset
- **aggsum** Compute sub-total sums by class of a categorical variable
- **colmad**, **colmean**, **colmed**, **colnorm**, 
    **colstd**, **colsum**, **colvar**  Column-wise operations
- **colmeanskip**, **colstdskip**, **colsumskip**, 
    **colvarskip**: Column-wise operations allowing missing data
- **covm**, **corm** Weighted covariance and correlation matrices
- **cosv**, **cosm** Cosinus between vectors
- **default** Display the keyword arguments (with their default values) of a function
- **dummy** Build dummy table
- **euclsq**, **mahsq**, **mahsqchol** Distances (Euclidean, Mahalanobis) between rows of matrices
- **fblockscal_col, _frob, _mfa, _sd** Scale blocks
- **fcenter**, **fscale**, **fcscale** Column-wise centering and scaling of a matrix
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
- **mlev** Return the sorted levels of a vecor or a dataset 
- **mweight** Normalize a vector to sum to 1
- **mweightcla** Compute observation weights for a categorical variable, 
    given specified sub-total weights for the classes
- **nco**, **nro**, Nb. rows and columns of an object
- **normw** Weighted norm of a vector
- **plist** Print each element of a list
- **pnames** Return the names of the elements of an object
- **psize** Return the type and size of a dataset
- **pval** Compute p-value(s) for a distribution, an ECDF or vector

- **recod_catbydict** Replace a categorical variable by dictionnary levels
- **recod_catbyind** Replace a categorical variable by indexes of levels
- **recod_catbyint** Replace a categorical variable by integers
- **recod_catbylev** Replace a categorical variable by levels
- **recod_indbylev** Replace an index variable by levels
- **recod_numbyint** Replace a continuous variable by integers

- **rmcol** Remove the columns of a matrix or the components of a vector having indexes s
- **rmrow** Remove the rows of a matrix or the components of a vector having indexes s
- **rowmean**, **rownorm**, **rowstd**, **rowsum**, **rowvar**: Row-wise operations
- **rowmeanskip**, **rowstdskip**, **rowsumskip**, **rowvarskip**: Row-wise operations 
    allowing missing data
- **soft** Soft thresholding
- **softmax** Softmax function
- **sourcedir** Include all the files contained in a directory
- **ssq** Total inertia of a matrix
- **summ** Summarize the columns of a dataset
- **tab**, **tabdf**, **tabdupl** Tabulations for categorical variables
- **vcatdf** Vertical concatenation of a list of dataframes
- **wdist** Compute weights from distances
- Other **utility functions** in file `utility.jl`


