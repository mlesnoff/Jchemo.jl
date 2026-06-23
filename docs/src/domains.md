# Available methods

## DIMENSION REDUCTION AND MULTIVARIATE EXPLORATORY DATA ANALYSES

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
- **pcapp** Projection pursuit
- **pcaout** Outlierness

*Sparse* 
- **spca** Sparse PCA by regularized low rank matrix approximation (sPCA-rSVD) *Shen & Huang 2008*

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

*Wrapper to UMAP.jl**

- **umap**: Uniform manifold approximation and projection for 
    dimension reduction

### Factorial discrimination analysis (FDA)

- **fda** Eigen decomposition of the consensus "inter/intra"
- **fdasvd** Weighted SVD of the class centers

### Partial covariances

- **covsel** Variable (feature) selection from partial covariance (Covsel) *Roger et al. 2011*

### Multiblock

*2 blocks*
- **cca** Canonical correlation analysis (CCA and RCCA)
- **ccawold** CCA and RCCA - Wold (1984) Nipals algorithm  
- **plscan** Canonical partial least squares regression (Symmetric PLS)
- **plstuck** Tucker's inter-battery method of factor analysis (PLS-SVD)
- **rasvd** Redundancy analysis (RA), a.k.a PCA on instrumental variables (PCAIV)

*2 or more blocks* 
- **cpca** Consensus principal components analysis (CPCA, a.k.a MBPCA) by Nipals
- **comdim** Common components and specific weights analysis (CCSWA, a.k.a ComDim or HPCA)

*Utilities*
- **mblock** Make blocks from a matrix
- **mbconcat** Concatenation of multi-block X-data
- **rd** Redundancy coefficients between two matrices
- **rv** RV correlation coefficient

## REGRESSION

### Multiple linear regression (MLR)

- **mlr** QR polyalgorithm (matrix division operator '\')

*Anova-related*
- **aov1** One-factor ANOVA
- **manova** MANOVA
- **hotelling** Two-sample Hotelling's T-squared test
- **decompx** Decomposition of a matrix by orthogonal projection on experimental factors.
- **asca** ANOVA Simultaneous Component Analysis (ASCA)
- **emm** Estimated marginal means (EMMs)
- **waldtest**: Wald or F test for model coefficients
- **wilks**: Compute statistics for multivariate tests

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
- **rrr** Reduced rank regression (RRR), a.k.a  Redundancy analysis regression 

*Robust*
- **plsrout** Outlierness

*Sparse*
- **splsr** sPLSR *Lê Cao et al. 2008*
- **spcr**  sPCR *Shen & Huang 2008*

*Averaging PLSR models of different dimensionalities*
- **plsravg** PLSR-AVG

*Non linear*
- **kplsr** Non linear kernel (KPLSR) *Rosipal & Trejo 2001*
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
- **krr** Non linear kernel (KRR), a.k.a Least squares SVM (LS-SVMR)

### Locally weighted models

- **loessr** LOESS regression model -- With package Loess.jl

*kNN*
- **knnr** kNN weighted regression (kNNR)
- **lwmlr** kNN locally weighted MLR (kNN-LWMLR)
- **lwplsr** kNN locally weighted PLSR (kNN-LWPLSR)

*Averaging*
- **lwplsravg** kNN-LWPLSR-AVG 

*Prototype models*
- **protoplsr** Averaging PLSR models built on the neighborhood of prototype observations
- **protoclustplsr** Clustered PLSR

### Support vector machines

*Wrapper to LIBSVM.jl*

- **svmr** Epsilon-SVR (SVM-R)

### Trees 

*Wrapper to DecisionTree.jl*

- **treer** Single tree
- **rfr** Random forest

### Bagging

- **baggr** Generic function for bagging a regression model
- **sampbag** Sampling utility function for bagging

## DISCRIMINATION ANALYSIS (DA)

### Based on the prediction of the Y-dummy table

*Linear*
- **mlrda** MLR-DA
- **plsrda** PLSR-DA, a.k.a usual PLSDA
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

### Locally weighted models

- **knnda** kNN-DA (Vote within neighbors)
- **lwmlrda** kNN locally weighted MLR-DA (kNN-LWMLR-DA)
- **lwplsrda** kNN Locally weighted PLSR-DA (kNN-LWPLSR-DA)
- **lwplslda** kNN Locally weighted PLS-LDA (kNN-LWPLS-LDA)
- **lwplsqda** kNN Locally weighted PLS-QDA (kNN-LWPLS-QDA, with continuum)

### Support vector machines 

*Wrapper to LIBSVM.jl*

- **svmda** C-SVC (SVM-DA)

### Trees 

*Wrapper to DecisionTree.jl*

- **treeda** Single tree
- **rfda** Random forest

## ONE-CLASS CLASSIFICATION (OCC)

### From Stahel-Donoho

- **occstah** Stahel-Donoho outlierness

### From a PCA or PLS score space

- **occsd** Score distance (SD)
- **occod** Orthogonal distance (OD) 
- **occsdod** SD-OD consensus
- **occdds**  SD2-OD2 consensus (with the DD-Simca approach)

### From kNN distance 

- **occknn**: kNN distance-based outlierness
- **occlknn**: Local kNN distance-based outlierness

### Utilities (unsupervised)

- **outstah** Stahel-Donoho outlierness
- **outeucl**: Outlierness from Euclidean distances to center
- **pcout**: Pcout algorithm for outlier identification in high dimensions *Filzmoser et al. 2008*
- **outsd**, **outod**, **outsdod**: Outlierness from PCA/PLS distances (SD, OD and consensus SD-OD)
- **outknn**: kNN distance-based outlierness
- **outlknn**: Local kNN distance-based outlierness

## DISTRIBUTIONS

- **dmnorm** Normal probability density estimation
- **dmnormlog** Logarithm of the normal probability density estimation
- **dmkern** Gaussian kernel density estimation (KDE)
- **pval** Compute p-value(s) for a distribution, a vector or an ECDF
- **out** Return if elements of a vector are strictly outside of a given range

## VARIABLE IMPORTANCE

- **vip** Variable importance on projections (VIP)
- **viperm!** Variable importance by direct permutations
- **isel!** Interval variable selection (e.g., Interval PLSR)

## TUNING MODELS

### Test-set validation

- **gridscore** Compute an error rate over a grid of parameters
  
### Cross-validation (CV)

- **gridcv** Compute an error rate over a grid of parameters
- **predictcv** Return the data and predictions per replication and segment from 
    a cross-validated model

### Utilities

- **mpar** Expand a grid of parameter values
- **segmkf** Build segments for K-fold CV
- **segmts** Build segments for test-set validation

### Performance scores

*Regression*
- **ssr** SSR
- **msep** MSEP
- **rmsep**, **rmseprel**, **rmsepstand** RMSEPs
- **mae** MAE
- **sep** SEP_c
- **bias** Bias
- **cor2** Squared correlation coefficient
- **r2** R2
- **rpd**, **rpdr** Ratios of performance to deviation
- **mse** Summary for regression
- **conf** Confusion matrix

*Discrimination*
- **errp** Classification error rate
- **merrp** Mean intra-class classification error rate

*Model dimensionality*
- **aicplsr** AIC and Cp for PLSR
- **selwold** Wold's criterion to select dimensionality in LV models (e.g., PLSR)

## DATA PROCESSING

### Checking

- **finduniq** Find the first indexes of a vector making unique the levels in this vector
- **dupl** Find replicated rows in a dataset
- **tabdupl** Tabulate duplicated values in a vector
- **findmiss** Find rows with missing data in a dataset

### Pre-processing

- De-trend transformation (baseline correction)
    - **detrend_pol** Polynomial linear regression
    - **detrend_lo** LOESS
    - **detrend_asls** Asymmetric least squares (ASLS)
    - **detrend_airpls** Adaptive iteratively reweighted penalized least squares (AIRPLS)
    - **detrend_arpls** Asymmetrically reweighted penalized least squares smoothing (ARPLS)
- **msc** : Multiplicative scatter correction
- **emsc** : Extended (polynomial) multiplicative scatter correction
- **snv** Standard-normal-deviation transformation
- **snorm** Row-wise norming
- **fdif** Finite differences
- **mavg** Smoothing by moving average
- **savgk**, **savgol** Savitsky-Golay filtering

*Centering-Scaling*
- **center** Column centering
- **scale** Column scaling
- **cscale** Column centering and scaling

- **blockscal** Scaling of multiblock data
- **fblockscal_col, _frob, _mfa, _sd** Scale blocks

*Remove gaps*
- **rmgap** Remove vertical gaps in spectra, e.g., for ASD NIR data

*Interpolation*
- **interpl** Sampling spectra by interpolation -- From DataInterpolations.jl

### Calibration transfer

- **difmean** Compute a detrimental matrix (for calibration transfer) by difference of two matrix-column means
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
- **plotxy** 2-D scatter plot of x-y data
- **plotxyz** 3-D scatter plot of x-y-z data
- **plotlv** Matrix of 2-D plots of successive latent variables (PCA, PLS, etc.)
- **plotgrid** Plot error/performance rates of a model
- **plotconf** Plot confusion matrix

## MODELS AND PIPELINES

- **model** Build a model
- **pip** Build a pipeline of models

## UTILITIES

*Macros*
- **@head** Display the first rows of a dataset
- **@pmod** Shortcut for function **parentmodule**
- **@names** Return the names of the sub-objects contained in a object
- **@pars** Display the keyword arguments (with their default values) of a function
- **@plist** Display each element of a list
- **@type** Display the type and size of a dataset

*Summarize a datase*
- **summ** Summarize the columns
- **aggstat** Column-wise statistics by group 
- **aggmean** Column-wise means by group
- **aggsumv** Sum by group of a categorical variable

*Tables*
- **tab**, **tabdupl** Tabulations for categorical variables
- **tabcont** Tabulate a continuous variable
- **mbin** Build histogram-bin intervals

*Weights*
- **pweight** Build an object of type 'StatsBase.ProbabilityWeights', with values summing to 1
- **pweightcla** Compute observation weights for a categorical variable, given specified sub-total weights for the classes
- Weights from distances
    - **wdis** Different functions to compute weights from distances
    - **wtal** Compute weights from distances using the 'talworth' distribution
    - **winvs** Compute weights from distances using an inverse scaled exponential function

*Recoding*
- Vector
    - **dummy** Build dummy table from a categorical variable
    - **recod_catbydict** Recode a categorical variable by levels defined in a dictionnary
    - **recod_catbyind** Recode a categorical variable by indexes of sorted levels
    - **recod_catbyind2** Recode a categorical variable by successive integer indexes
    - **recod_catbylev** Recode a categorical variable by levels
    - **recod_indbylev** Recode an index variable by levels
    - **recod_contbylev** Recode a quantitative variable by successive levels
    - **recod_miss** Declare data as missing in a dataset
- Data
    - **expand_tab2d** Expand a 2-D contingency table to a dataframe of two categorical variables
    - **expand_grid** Build a dataframe with all the combinations of the entered parameter values
    - **convertdf** Convert the columns of a dataframe to given types

*Operations on a vector*
- **sumv**, **meanv**, **normv**, **norm2v**, **stdv**, **varv**, **medv**,  **madv**, **iqrv**: Statistics
- **boxcox**, **boxcox_transf**: Box-Cox power transformation to normalize a variable

*Operations on two vectors*
- **cosv** Cosinus 
- **covv** Covariances
- **corv** Correlations

*Operations on a matrix*
- **cosm** Cosinus matrix
- **covm** Covariances matrix
- **corm** Correlations matrix
- **frob**, **frob2** Frobenius norm
- **matB**, **matW**, **matWc** Between- and within-class covariance matrices

- Column-wise
    - **nco** Nb. columns 
    - **colsum** Sum
    - **colmean** Mean
    - **colnorm** Norm
    - **colnorm2** Squared norm
    - **colstd** Standard deviation (uncorrected)
    - **colvar** Variance (uncorrected)
    - **colmed** Median
    - **colmad** Median absolute deviation (MAD)
    - Allow missing data
        - **colsumskip**, **colmeanskip**, **colstdskip**, **colvarskip** 

 - Row-wise
    - **nro** Nb. rows 
    - **rowsum** Sum
    - **rowmean** Mean
    - **rownorm** Norm
    - **rownorm2** Squared norm
    - **rowstd** Standard deviation (uncorrected)
    - **rowvar** Variance (uncorrected)
    - Allow missing data
        - **rowsumskip**, **rowmeanskip**, **rowstdskip**, **rowvarskip** 

*Transformation of a amatrix*
- **fweightr** Weight each row
- **fweightc** Weight each column
- **fcenter**, **fscale**, **fcscale** Column-wise centering and scaling
- **rmcol**, **rmrow** Remove columns and rows
- **fconcat** Concatenate multiblock data

*Others*
- **eucl2**, **mah2**, **mah2chol** Distances (Euclidean, Mahalanobis) between rows of matrices
- **findmax_cla** Find the most occurent level in a categorical variable
- **getknn** Find nearest neighbors between rows of matrices
- **krbf, kpol** Build kernel Gram matrices
- **locw** Working function for local (kNN) models
- **mad** Median absolute deviation (not exported)
- **mlev** Return the sorted levels of an array or dataset 
- **parsemiss** Parsing a string vector allowing missing data
- **pval** Compute p-value(s) from a distribution, an ECDF or a vector
- **thresh_soft**, **thresh_hard** Thresholding functions
- **softmax** Softmax function
- **sourcedir** Include all the files contained in a directory
- **vcatdf** Vertical concatenation of a list of dataframes
- Other **utility functions** in files `_util.jl`

