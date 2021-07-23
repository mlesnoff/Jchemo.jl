#### <span style="color:#1589F0"> **PCA** </span>

- ***pcaeigen*** Eigen decomposition
- ***pcaeigenk*** Eigen decomposition for wide matrices (kernel form)
- ***pcasvd*** SVD decomposition
- ***kpca*** Non linear kernel PCA  (KPCA) (Scholkopf et al. 2002)

*Utility (works also for PLS)*
- ***scordis*** Score distances (SDs) for a score space
- ***odis*** Orthogonal distances (ODs) for a score space
- ***xfit*** Matrix fitting 
- ***xresid*** Residual matrix  

#### <span style="color:#1589F0;"> **REGRESSION** </span>

#### **Linear (LMR)**

- ***lmr*** QR algorithm
- ***lmrchol*** Normal equations and Choleski factorization
- ***lmrpinv*** Pseudo-inverse
- ***lmrpinv_n*** Normal equations and pseudo-inverse
- ***lmrvec*** Univariate X

*Ill-conditionned* 

- ***cglsr*** Conjugate gradient for the Normal equations (CGLS)

#### **Partial least squares (PLSR)**

- ***plskernel*** "Improved kernel #1" (Dayal & McGregor 1997)
- ***plsnipals*** NIPALS
- ***kplsr*** Non linear kernel PLSR (KPLSR) (Rosipal & Trejo 2001)
- ***dkplsr*** Direct KPLSR (Bennett & Embrechts 2003)
<!---
- ***plsnipals*** Nipals
- ***plsrannar*** Kernel version for wide matrices (Rannar et al. 1994)
-->

#### **Ridge (RR)**

- ***rr*** Pseudo-inverse
- ***rrchol*** Choleski factorization
- ***krr*** Non linear kernel RR (KRR), i.e. Least squares SVM (LS-SVMR)

<!---
#### Support vector machine
- ***svmr*** SVM regression (SVMR)
-->

#### **k-nearest-neighbors (kNN)**

- ***knnr*** kNNR
- ***lwplsr*** kNN Locally weighted PLSR (kNN-LWPLSR)

#### **Ensemblist methods**

*Averaging PLSR models with different numbers of LVs*

- ***plsr_agg*** PLSR-AGG
- ***lwplsr_agg*** KNN-LWPLSR-AGG 

*Bagging, Boosting*

- ***baggr*** Bagging

- ***boostr***, ***boostrw*** Adaptative boosting

- ***gboostr*** Gradient boosting

#### <span style="color:#1589F0"> DISCRIMINATION ANALYSIS </span>
  
<!---
#### Factorial discrimination analysis (FDA)

- ***fda*** Eigen decomposition of the compromise "inter/intra"
- ***fdasvd*** Weighted SVD decomposition of the class centers

#### On predicted Y-dummy table

- ***lmrda*** DA on LMR prediction (LMR-DA)
- ***plsrda*** DA on PLSR prediction (PLSR-DA = common "PLSDA")
- ***kplsrda*** DA on KPLSR prediction (KPLSR-DA)
- ***rrda*** DA on RR prediction (RR-DA)
- ***krrda*** DA on KRR prediction (KRR-DA)

#### Probabilistic

- ***lda*** Linear discriminant analysis (LDA)
- ***qda*** Quadratic discriminant analysis (QDA)
- ***plslda*** LDA on PLS latent variables (LVs) (PLS-LDA)
- ***plsqda*** QDA on PLS LVs (PLS-QDA)

#### Support vector machine

- ***svmda*** SVMDA (= SVMC)

#### K-nearest-neighbors

- ***knnda*** KNN-DA
- ***lwplsrda*** KNN Locally weighted PLSR-DA (KNN-LWPLSR-DA)
- ***lwplslda*** KNN Locally weighted PLS-LDA/QDA (KNN-LWPLS-LDA/QDA)
-->

<!---
#### <span style="color:#1589F0"> **ENSEMBLIST METHODS** </span>
- ***plsrda_agg*** PLSRDA-AGG
- ***lwplsrda_agg*** KNN-LWPLSR-DA-AGG
- ***lwplslda_agg*** KNN-LWPLS-LDA-AGG
- ***lwplslda_agg*** KNN-LWPLS-QDA-AGG
-->

#### <span style="color:#1589F0"> TUNING MODELS </span>

#### **Grid**

- ***mpars*** Expand a grid of parameter values

#### **Validation**

- ***gridscore*** Any model
- ***gridscorelv*** Models with LVs (faster)
- ***gridscorelb*** Models with ridge parameter (faster)
  
#### **Cross-validation (CV)**

<!---
- ***gridcv*** Any model
- ***gridcvlv*** Models with LVs (faster)
- ***gridcvlb*** Models with ridge parameter (faster)  
-->

<!---
- ***segmkf*** Building segments for K-fold CV
- ***segmts*** Building segments for test-set CV
-->

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

#### **Heuristic**  

<!---  
- ***selwold*** Wold's criterion for models with LVs  
-->

#### <span style="color:#1589F0"> **SELECTION OF VARIABLES** </span>

<!---
- ***covsel*** COVSEL algorithm (Roger et al. 2011)
-->

#### <span style="color:#1589F0"> **DATA MANAGEMENT** </span>

#### **Pre-processing**

- ***snv*** Standard-normal-deviation transformation
- ***detrend*** Polynomial detrend
- ***fdif*** Finite differences
- ***mavg***, ***mavg_runmean*** Smoothing by moving average
- ***savgk***, ***savgol*** Savitsky-Golay filtering
<!--- 
- ***xinterp*** Resampling of spectra by interpolation methods
- ***gaprm** Remove vertical gaps in spectra (e.g. for ASD)
- **eposvd** Pre-processing data by external parameter orthogonalization (EPO; Roger et al 2003) 
-->

#### **Sampling**

<!---
- ***sampks*** Kennard-Stone sampling 
- ***sampdp*** Duplex sampling 
- ***sampclas*** Within-class (stratified) sampling
-->

#### **Checking**

<!---
- ***checkna*** Find and count NA values in a data set
- ***plotxna*** Plotting missing data in a matrix
- ***checkdupl*** Find duplicated row observations between two data sets 
- ***rmdupl*** Remove duplicated row observations between two data sets
-->

#### **Summary**

<!---
- ***aggmean*** Centers of classes
- ***dtagg*** Summary statistics with data subsets
- ***summ*** Summary of the quantitative variables of a data set
-->

#### **Multi-block**

<!---
- ***mblocks*** Makes a list of blocks
- ***hconcat*** Horizontal block concatenation 
- ***blockscal*** Block autoscaling
-->

#### **Datasets**

<!---
- ***asdgap** ASD spectra with vertical gaps
- ***cassav*** Tropical shrubs
- ***forages*** Tropical forages
- ***octane*** Gazoline "octane" dataset
- ***ozone*** Los Angeles "ozone" pollution (1976) dataset
-->

#### <span style="color:#1589F0"> **GRAPHICS** </span>

<!---
- ***plotsp*** Plotting spectra, loadings, or more generally row observations of a data set
- ***plostsp1*** Same as  ***plotsp*** but one-by-one row
- ***plotxy*** 2-d scatter plot
- ***plotjit*** Jittered plot
- ***plotscore*** Plotting error rates of prediction models
-->

#### <span style="color:#1589F0"> **UTILITY** </span>

<!---
- ***dmnorm*** Multivariate normal probability density
- ***dummy*** Dummy table
- ***euclsq***, ***euclsq_mu** Euclidean distance matrices
- ***mahsq***, ***mahsq_mu** Mahalanobis distance matrices
- ***getknn*** KNN selection
- ***krbf***, ***kpol***, ***ktanh*** Gram matrices for different kernels
- ***locw*** Working function for locally weighted models
- ***matB***, ***matW*** Between and within covariance matrices
- ***sourcedir*** Source every R functions in a directory
- ***wdist*** Weights for distances
- Additional working functions in file **zfunctions.R**
-->
