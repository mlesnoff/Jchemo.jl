### <span style="color:#1589F0"> **PCA** </span>

- ***pcaeigen*** Eigen decomposition
- ***pcaeigenk*** Eigen decomposition for wide matrices (kernel form)
- ***pcasvd*** SVD decomposition
- ***kpca*** Non linear kernel PCA  (KPCA) *Scholkopf et al. 2002*

*Utility (works also for PLS)*
- ***scordis*** Score distances (SDs) for a score space
- ***odis*** Orthogonal distances (ODs) for a score space
- ***xfit*** Matrix fitting 
- ***xresid*** Residual matrix 

### <span style="color:#1589F0"> **RANDOM PROJECTIONS** </span>

- ***rpmat_gauss*** Gaussian random projection matrix 
- ***rpmat_li*** Sparse random projection matrix 
- ***rp*** Random projection

### <span style="color:#1589F0;"> **REGRESSION** </span>

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
- ***plsrosa*** ROSA *Liland et al. 2016*
- ***kplsr*** Non linear kernel PLSR (KPLSR) *Rosipal & Trejo 2001*
- ***dkplsr*** Direct KPLSR *Bennett & Embrechts 2003*
- ***plsr_agg*** Averaging PLSR models with different numbers of LVs (PLSR-AGG)

*Utility*
- ***aicplsr*** AIC and Cp for PLSR

#### **Principal component (PCR)**

- ***pcr*** PCR by SVD factorization

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

#### Variable selection

- ***covsel*** CovSel (Roger et al. 2011)
- ***iplsr*** Interval PLSR (iPLS) (Nørgaard et al. 2000)

### <span style="color:#1589F0"> **DISCRIMINATION ANALYSIS (DA)** </span>

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

### <span style="color:#1589F0;"> **MULTIBLOCK** </span>

*Regression*

- ***rosaplsr*** Multiblock PLSR with the ROSA algorithm

*Utility*

- ***blockscal*** Autoscale a list of blocks (e.g. for MB-PLS)
- ***mblocks*** Make blocks from a matrix
- ***rv*** RV correlation coefficient

### <span style="color:#1589F0"> **TUNING MODELS** </span>

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

### <span style="color:#1589F0"> **DATA MANAGEMENT** </span>

### **Checking**

- ***checkdupl*** Finding replicated rows in a dataset
- ***checkmiss*** Finding rows with missing data in a dataset

#### **Pre-processing**

- ***detrend*** Polynomial detrend
- ***eposvd*** External parameter orthogonalization (EPO)
- ***fdif*** Finite differences
- ***interpl*** Sampling signals by intrerpolation -- From DataInterpolations.jl
- ***interpl_mon*** Sampling signals by monotonic intrerpolation -- From Interpolations.jl
- ***mavg***, ***mavg_runmean*** Smoothing by moving average
- ***rmgap*** Remove vertical gaps in spectra, e.g. for ASD NIR data
- ***savgk***, ***savgol*** Savitsky-Golay filtering
- ***snv*** Standard-normal-deviation transformation

#### **Sampling observations**

- ***sampks*** Kennard-Stone sampling 
- ***sampdp*** Duplex sampling 
- ***sampsys*** Systematic sampling
- ***sampclas*** Stratified sampling

### <span style="color:#1589F0"> **UTILITIES** </span>

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
- ***plsrannar*** Kernel version for wide matrices (Rannar et al. 1994)
- ***simpls***
-->

<!---  
#### **Heuristic**  
- ***selwold*** Wold's criterion for models with LVs  
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
- ***plotxna*** Plotting missing data in a matrix
-->

<!---
#### **Datasets**
- ***asdgap** ASD spectra with vertical gaps
- ***cassav*** Tropical shrubs
- ***forages*** Tropical forages
- ***octane*** Gazoline "octane" dataset
- ***ozone*** Los Angeles "ozone" pollution (1976) dataset
-->
