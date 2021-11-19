### <span style="color:#1589F0"> **1. PCA** </span>

- ***pcaeigen*** Eigen decomposition
- ***pcaeigenk*** Eigen decomposition for wide matrices (kernel form)
- ***pcasvd*** SVD decomposition
- ***kpca*** Non linear kernel PCA  (KPCA) *Scholkopf et al. 2002*

*Utility (works also for PLS)*
- ***scordis*** Score distances (SDs) for a score space
- ***odis*** Orthogonal distances (ODs) for a score space
- ***xfit*** Matrix fitting 
- ***xresid*** Residual matrix 

### <span style="color:#1589F0;"> **2. REGRESSION** </span>

#### **Linear models**

*Anova*

- ***aov1*** One factor ANOVA

*Multiple linear regression (MLR)*

- ***mlr*** QR algorithm
- ***mlrchol*** Normal equations and Choleski factorization
- ***mlrpinv*** Pseudo-inverse
- ***mlrpinv_n*** Normal equations and pseudo-inverse
- ***mlrvec*** Simple linear regression (Univariate x)

*Ill-conditionned* 

- ***cglsr*** Conjugate gradient for the least squares normal equations (CGLS)

#### **Partial least squares (PLSR)**

- ***plskern*** "Improved kernel #1" *Dayal & McGregor 1997*
- ***plsnipals*** NIPALS
- ***kplsr*** Non linear kernel PLSR (KPLSR) *Rosipal & Trejo 2001*
- ***dkplsr*** Direct KPLSR *Bennett & Embrechts 2003*
- ***aicplsr*** AIC and Cp for PLSR
- ***plsr_agg*** Averaging PLSR models with different numbers of LVs (PLSR-AGG)

#### **Ridge (RR, KRR)**

- ***rr*** Pseudo-inverse
- ***rrchol*** Choleski factorization
- ***krr*** Non linear kernel RR (KRR) = Least squares SVM (LS-SVMR)

#### **Local models**

- ***knnr*** kNNR
- ***lwplsr*** kNN Locally weighted PLSR (kNN-LWPLSR)
- ***lwplsr_agg*** kNN-LWPLSR-AGG 
- ***cplsr_agg*** Clustered PLSR-AGG

#### **Support vector machine (SVMR)** -- from LIBSVM.jl
- ***svmr*** Epsilon-SVM regression

#### **Trees** -- from XGBoost.jl

- ***treer_xgb*** Single tree
- ***rfr_xgb*** Random forest
- ***xgboostr*** XGBoost
- ***vimp_xgb*** Variable importance (Works also for DA models)

#### **Bagging**

- ***baggr*** Bagging 
- ***baggr_oob*** Out-of-bag error rate
- ***baggr_vi*** Variance importance (permutation method)

### <span style="color:#1589F0"> **3. DISCRIMINATION ANALYSIS (DA)** </span>

#### Factorial discrimination analysis (FDA)

- ***fda*** Eigen decomposition of the compromise "inter/intra"
- ***fdasvd*** Weighted SVD decomposition of the class centers

#### DA based on predicted Y-dummy table

- ***mlrda*** Based on MLR predictions (MLR-DA)
- ***plsrda*** Based on PLSR predictions (PLSR-DA; = common "PLSDA")
- ***kplsrda*** DA on KPLSR predictions (KPLSR-DA)
- ***rrda*** DA on RR predictions (RR-DA)
- ***krrda*** DA on KRR predictions (KRR-DA)
- ***plsrda_agg*** Averaging PLSR-DA models with different numbers of LVs (PLSR-DA-AGG)

#### Probabilistic

- ***dmnorm*** Normal probability density of multivariate data
- ***lda*** Linear discriminant analysis (LDA)
- ***qda*** Quadratic discriminant analysis (QDA)
- ***plslda*** LDA on PLS latent variables (PLS-LDA)
- ***plsqda*** QDA on PLS latent variables (PLS-QDA)
- ***plslda_agg*** Averaging PLS-LDA models with different numbers of LVs (PLS-LDA-AGG)
- ***plsqda_agg*** Averaging PLS-QDA models with different numbers of LVs (PLS-QDA-AGG)

#### **Local models**

- ***knnda*** kNN-DA (Vote within neighbors)
- ***lwplsrda*** kNN Locally weighted PLSR-DA (kNN-LWPLSR-DA)
- ***lwplslda***, ***lwplsqda*** kNN Locally weighted PLS-LDA/QDA (kNN-LWPLS-LDA/QDA)
- ***lwplslda_agg***, ***lwplsqda_agg*** Averaging kNN-LWPLS-LDA/QDA models with different numbers of LVs (kNN-LWPLS-LDA/QDA-AGG)

#### **Support vector machine (SVM-DA)** -- from LIBSVM.jl
- ***svmda*** C-SVM discrimination

#### **Trees** -- from XGBoost.jl

- ***treeda_xgb*** Single tree
- ***rfda_xgb*** Random forest
- ***xgboostda*** XGBoost

### <span style="color:#1589F0"> **4. TUNING MODELS** </span>

#### **Grid**

- ***mpars*** Expand a grid of parameter values

#### **Validation**

- ***gridscore*** Any model
- ***gridscorelv*** Models with LVs (faster)
- ***gridscorelb*** Models with ridge parameter (faster)
  
#### **Cross-validation (CV)**

- ***segmts*** Building segments for test-set validation
- ***segmkf*** Building segments for K-fold CV
- ***gridcv*** Any model
- ***gridcvlv*** Models with LVs (faster)
- ***gridcvlb*** Models with ridge parameter (faster)  

#### **Performance scores**

- ***ssr*** SSR
- ***msep*** MSEP
- ***rmsep*** RMSEP
- ***sep*** SEP
- ***bias*** Bias
- ***r2*** R2
- ***cor2*** Squared correlation coefficient
- ***rpd***, ***rpdr*** Ratio of performance to deviation
- ***mse*** Summary for regression
- ***err*** Classification error rate

### <span style="color:#1589F0"> **5. DATA MANAGEMENT** </span>

#### **Pre-processing**

- ***snv*** Standard-normal-deviation transformation
- ***detrend*** Polynomial detrend
- ***fdif*** Finite differences
- ***mavg***, ***mavg_runmean*** Smoothing by moving average
- ***savgk***, ***savgol*** Savitsky-Golay filtering
- ***eposvd*** External parameter orthogonalization (EPO)
- ***rmgap*** Remove vertical gaps in spectra, e.g. for ASD NIR data
- ***interpl*** Resampling of signals by spline interpolation.

#### **Sampling observations**

- ***sampks*** Kennard-Stone sampling 
- ***sampdp*** Duplex sampling 
- ***sampsys*** Systematic sampling
- ***sampclas*** Stratified sampling

### <span style="color:#1589F0"> **6. UTILITY** </span>

- ***aggstat*** Compute column-wise statistics (e.g. mean), by group
- ***center***, ***scale*** Column-wise centering and scaling of a matrix
- ***colmeans***, ***colvars*** Weighted column means and variances
- ***dummy*** Build dummy table
- ***euclsq***, ***mahsq***, ***mahsqchol*** Euclidean and Mahalanobis distances between rows of matrices
- ***getknn*** Find nearest neighbours between rows of matrices
- ***iqr*** Interval inter-quartiles
- ***krbf, kpol*** Build kernel Gram matrices
- ***locw*** Working function for local (kNN) models
- ***mad*** Median absolute deviation
- ***matcov***, ***matB***, ***matW*** Covariances matrices
- ***recodnum2cla*** Recode a continuous variable to classes
- ***sourcedir*** Include all the files contained in a directory
- ***summ*** Summarize the columns of a dataset
- ***tab***, ***tabn*** Univariate tabulation 
- ***wdist*** Compute weights from distances
- Other functions in file `utility.jl`

#### **Plotting**

- ***plotsp*** Plotting spectra



<!---
#### **Other variable importance methods**

- ***vimp_perm*** 
- ***vimp_chisq_r*** 
- ***vimp_aov_r*** 
-->

<!---
#### Support vector machine
- ***svmda*** SVMDA (= SVMC)
-->

<!---
- ***plsrannar*** Kernel version for wide matrices (Rannar et al. 1994)
- ***simpls***
-->

<!---  
#### **Heuristic**  
- ***selwold*** Wold's criterion for models with LVs  
-->

<!---
### <span style="color:#1589F0"> **SELECTION OF VARIABLES** </span>
- ***covsel*** COVSEL algorithm (Roger et al. 2011)
-->

<!---
#### **Multi-block**
- ***mblocks*** Makes a list of blocks
- ***hconcat*** Horizontal block concatenation 
- ***blockscal*** Block autoscaling
-->

<!---
### <span style="color:#1589F0"> **GRAPHICS** </span>
- ***plostsp1*** Same as  ***plotsp*** but one-by-one row
- ***plotxy*** 2-d scatter plot
- ***plotjit*** Jittered plot
- ***plotscore*** Plotting error rates of prediction models
-->

<!---
#### **Checking**
- ***checkna*** Find and count NA values in a data set
- ***plotxna*** Plotting missing data in a matrix
- ***checkdupl*** Find duplicated row observations between two data sets 
- ***rmdupl*** Remove duplicated row observations between two data sets
-->

<!---
#### **Datasets**
- ***asdgap** ASD spectra with vertical gaps
- ***cassav*** Tropical shrubs
- ***forages*** Tropical forages
- ***octane*** Gazoline "octane" dataset
- ***ozone*** Los Angeles "ozone" pollution (1976) dataset
-->
