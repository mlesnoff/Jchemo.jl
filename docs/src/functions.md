### <span style="color:#1589F0"> **PCA** </span>

- ***kpca*** Non linear kernel  (KPCA) *Scholkopf et al. 2002*
- ***pcaeigen*** Eigen decomposition
- ***pcaeigenk*** Eigen decomposition for wide matrices (kernel form)
- ***pcasvd*** SVD decomposition

*Utility (works also for PLS)*
- ***odis*** Orthogonal distances (ODs) for a score space
- ***scordis*** Score distances (SDs) for a score space
- ***xfit*** Matrix fitting 
- ***xresid*** Residual matrix 

*Multiblock*
- ***blockscal[_col, _frob, _mfa, _sd]*** Scaling blocks
- ***lg*** Lg coefficient
- ***mbpca_cons*** Consensus (CPCA, MBPCA)
- ***mbpca_comdim_s*** Common components and specific weights (CCSWA, ComDim)
- ***rv*** RV correlation coefficient

### <span style="color:#1589F0"> **RANDOM PROJECTIONS** </span>

- ***rp*** Random projection
- ***rpmat_gauss*** Gaussian random projection matrix 
- ***rpmat_li*** Sparse random projection matrix 

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

- ***dkplsr*** Direct non linear kernel (DKPLSR) *Bennett & Embrechts 2003*
- ***kplsr*** Non linear kernel (KPLSR) *Rosipal & Trejo 2001*
- ***plskern*** "Improved kernel #1" *Dayal & McGregor 1997*
- ***plsnipals*** NIPALS
- ***plsrosa*** ROSA *Liland et al. 2016*
- ***plssimp*** SIMPLS *de Jong 1993*
- ***plsr_avg*** Averaging and stacking PLSR models with different numbers of LVs (PLSR-AVG)

*Utility*
- ***aicplsr*** AIC and Cp for PLSR

*Variable selection*
- ***covsel*** CovSel (Roger et al. 2011)
- ***iplsr*** Interval PLSR (iPLS) (Nørgaard et al. 2000)

*Multiblock*
- ***mbplsr*** Multiblock (MBPLSR; concatenated autoscaled blocks)
- ***mbplsr_rosa*** ROSA *Liland et al. 2016*
- ***mbplsr_so*** Sequentially orthogonalized (SO-PLSR) 

#### **Principal component (PCR)**

- ***pcr*** SVD factorization

#### **Ridge (RR, KRR)**

- ***krr*** Non linear kernel (KRR) = Least squares SVM (LS-SVMR)
- ***rr*** Pseudo-inverse (RR)
- ***rrchol*** Choleski factorization (RR)

#### **Local models**

- ***cplsr_avg*** Clustered PLSR-AVG
- ***knnr*** kNNR
- ***lwplsr*** kNN Locally weighted PLSR (kNN-LWPLSR)
- ***lwplsr_avg*** kNN-LWPLSR-AVG 
- ***lwplsr_s*** kNN-LWPLSR with preliminary dimension reduction

#### **Support vector machine (SVMR)** -- from LIBSVM.jl
- ***svmr*** Epsilon-SVM regression

#### **Trees** -- from XGBoost.jl

- ***rfr_xgb*** Random forest
- ***treer_xgb*** Single tree
- ***vimp_xgb*** Variable importance (Works also for DA models)
- ***xgboostr*** XGBoost

#### **Bagging**

- ***baggr*** Bagging 
- ***baggr_oob*** Out-of-bag error rate
- ***baggr_vi*** Variance importance (permutation method)

### <span style="color:#1589F0"> **DISCRIMINATION ANALYSIS (DA)** </span>

#### Factorial discrimination analysis (FDA)

- ***fda*** Eigen decomposition of the compromise "inter/intra"
- ***fdasvd*** Weighted SVD decomposition of the class centers

#### DA based on predicted Y-dummy table

- ***kplsrda*** On KPLSR predictions (KPLSR-DA)
- ***krrda*** On KRR predictions (KRR-DA)
- ***mlrda*** On MLR predictions (MLR-DA)
- ***plsrda*** On PLSR predictions (PLSR-DA; = common "PLSDA")
- ***plsrda_avg*** Averaging PLSR-DA models with different numbers of LVs (PLSR-DA-AVG)
- ***rrda*** On RR predictions (RR-DA)

#### Probabilistic

- ***dmnorm*** Normal probability density of multivariate data
- ***lda*** Linear discriminant analysis (LDA)
- ***plslda*** LDA on PLS latent variables (PLS-LDA)
- ***plslda_avg*** Averaging PLS-LDA models with different numbers of LVs (PLS-LDA-AVG)
- ***plsqda*** QDA on PLS latent variables (PLS-QDA)
- ***plsqda_avg*** Averaging PLS-QDA models with different numbers of LVs (PLS-QDA-AVG)
- ***qda*** Quadratic discriminant analysis (QDA)

#### **Local models**

- ***knnda*** kNN-DA (Vote within neighbors)
- ***lwplsrda*** kNN Locally weighted PLSR-DA (kNN-LWPLSR-DA)
- ***lwplslda***, ***lwplsqda*** kNN Locally weighted PLS-LDA/QDA (kNN-LWPLS-LDA/QDA)
- ***lwplslda_avg***, ***lwplsqda_avg*** Averaging kNN-LWPLS-LDA/QDA models with different numbers of LVs (kNN-LWPLS-LDA/QDA-AVG)

#### **Support vector machine (SVM-DA)** -- from LIBSVM.jl
- ***svmda*** C-SVM discrimination

#### **Trees** -- from XGBoost.jl

- ***treeda_xgb*** Single tree
- ***rfda_xgb*** Random forest
- ***xgboostda*** XGBoost

### <span style="color:#1589F0"> **TUNING MODELS** </span>

#### **Grid**

- ***mpar*** Expand a grid of parameter values

#### **Validation**

- ***gridscore*** Any model
- ***gridscorelv*** Models with LVs (faster)
- ***gridscorelb*** Models with ridge parameter (faster)
  
#### **Cross-validation (CV)**

- ***gridcv*** Any model
- ***gridcvlv*** Models with LVs (faster)
- ***gridcvlb*** Models with ridge parameter (faster)  
- ***gridcv_mb*** Multiblock models 
- ***gridcvlv_mb*** Multiblock models with LVs 
- ***segmkf*** Building segments for K-fold CV
- ***segmts*** Building segments for test-set validation

#### **Performance scores**

- ***bias*** Bias
- ***cor2*** Squared correlation coefficient
- ***err*** Classification error rate
- ***mse*** Summary for regression
- ***msep*** MSEP
- ***rmsep*** RMSEP
- ***sep*** SEP
- ***r2*** R2
- ***rpd***, ***rpdr*** Ratio of performance to deviation
- ***ssr*** SSR

### <span style="color:#1589F0"> **DATA MANAGEMENT** </span>

#### **Calibration transfert**

- ***caltransf_ds*** : Direct standardization (DS).
- ***caltransf_pds*** : Piecewise direct standardization (PDS).

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

- ***sampclas*** Stratified sampling
- ***sampdp*** Duplex sampling 
- ***sampks*** Kennard-Stone sampling 
- ***sampsys*** Systematic sampling

### <span style="color:#1589F0"> **UTILITIES** </span>

- ***aggstat*** Compute column-wise statistics (e.g. mean), by group
- ***center***, ***scale*** Column-wise centering and scaling of a matrix
- ***colmean***, ***colnorm2***, ***colstd***, ***colsum***, ***colvar***  Column-wise operations
- ***covm***, ***corm*** Covariance and correlation matrices
- ***datasets*** Datasets available in the package
- ***dummy*** Build dummy table
- ***euclsq***, ***mahsq***, ***mahsqchol*** Distances (Euclidean, Mahalanobis) between rows of matrices
- ***fweight*** Compute weights from distances
- ***getknn*** Find nearest neighbours between rows of matrices
- ***iqr*** Interval inter-quartiles
- ***krbf, kpol*** Build kernel Gram matrices
- ***locw*** Working function for local (kNN) models
- ***mad*** Median absolute deviation
- ***matB***, ***matW*** Between- and within-covariance matrices
- ***mblock*** Make blocks from a matrix
- ***mweight*** Normalize a vector to sum to 1.
- ***nco***, ***nro***, Nb. rows and colmuns of an object.
- ***norm2*** Squared norm of a vector
- ***recodnum2cla*** Recode a continuous variable to classes
- ***rowmean***, ***rowstd***, ***rowsum*** Row-wise operations
- ***sourcedir*** Include all the files contained in a directory
- ***ssq*** Total inertia of a matrix
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
