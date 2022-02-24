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

*Multiblock*
- ***blockscal[_col, _frob, _mfa, _sd]*** Scaling blocks (for multi-block analyses)
- ***lg*** Lg coefficient
- ***mbpca_cons*** Consensus PCA (CPCA, MBPCA)
- ***mbpca_comdim_s*** Common components and specific weights analysis (CCSWA, ComDim)
- ***rv*** RV correlation coefficient

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
- ***plssimp*** SIMPLS *de Jong 1993*
- ***kplsr*** Non linear kernel PLSR (KPLSR) *Rosipal & Trejo 2001*
- ***dkplsr*** Direct KPLSR *Bennett & Embrechts 2003*
- ***plsr_avg*** Averaging PLSR models with different numbers of LVs (PLSR-AVG)

*Utility*
- ***aicplsr*** AIC and Cp for PLSR

*Variable selection*
- ***covsel*** CovSel (Roger et al. 2011)
- ***iplsr*** Interval PLSR (iPLS) (Nørgaard et al. 2000)

*Multiblock*
- ***mbplsr*** Multiblock PLSR (PLSR on concatenated autoscaled blocks)
- ***mbplsr_rosa*** Multiblock ROSA PLSR *Liland et al. 2016*
- ***mbplsr_so*** Sequentially orthogonalized PLSR (SO-PLSR) 

#### **Principal component (PCR)**

- ***pcr*** PCR by SVD factorization

#### **Ridge (RR, KRR)**

- ***rr*** Pseudo-inverse
- ***rrchol*** Choleski factorization
- ***krr*** Non linear kernel RR (KRR) = Least squares SVM (LS-SVMR)

#### **Local models**

- ***knnr*** kNNR
- ***lwplsr*** kNN Locally weighted PLSR (kNN-LWPLSR)
- ***lwplsr_avg*** kNN-LWPLSR-AVG 
- ***lwplsr_s*** kNN-LWPLSR with preliminary dimension reduction
- ***cplsr_avg*** Clustered PLSR-AVG

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
- ***plsrda_avg*** Averaging PLSR-DA models with different numbers of LVs (PLSR-DA-AVG)

#### Probabilistic

- ***dmnorm*** Normal probability density of multivariate data
- ***lda*** Linear discriminant analysis (LDA)
- ***qda*** Quadratic discriminant analysis (QDA)
- ***plslda*** LDA on PLS latent variables (PLS-LDA)
- ***plsqda*** QDA on PLS latent variables (PLS-QDA)
- ***plslda_avg*** Averaging PLS-LDA models with different numbers of LVs (PLS-LDA-AVG)
- ***plsqda_avg*** Averaging PLS-QDA models with different numbers of LVs (PLS-QDA-AVG)

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

- ***segmts*** Building segments for test-set validation
- ***segmkf*** Building segments for K-fold CV
- ***gridcv*** Any model
- ***gridcvlv*** Models with LVs (faster)
- ***gridcv_mb*** Multiblock models 
- ***gridcvlv_mb*** Multiblock models with LVs 
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
