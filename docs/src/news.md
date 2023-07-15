### NEWS for package **Jchemo**

### *Version 0.1.25*

- News
    - **Nsc**: Nearest shrunken centroids (and variable selection).
    - **softmax** Softmax function.

- Modifications
    - Code improvement.

### *Version 0.1.24*

- News
    - **soft**: Soft thresholding.

- Modifications
    - Code improvement.


### *Version 0.1.23*

- News
    - **dmnormlog**: Logarithm of the normal probability density estimation.
    - **rda**: Regularized discriminant analysis (RDA)

- Modifications
    - **matB**, **matW**, **lda**, **qda**: new argument 'weights'.
    - **qda**, **plsqda**, **lwplsqda**, **plsqdaavg**, **lwplsqdaavg**: new argument 'alpha' 
        (continuum from QDA toward LDA). 
    - **dmnorm**: new argument 'simpl'. 
    - Code improvement.

- Breaking changes
    - **checkdupl**, **checkmiss**: renamed to **dupl** and **miss**.
    - **fda**, **fdasvd** : argument 'pseudo' has been replaced by 'lb'
        (ridge regularization).

### *Version 0.1.22*

- News
    - **difmean** : Compute a detrimental matrix (for calibration transfer) by column 
        means difference.

- Modifications
    - Code improvement.

- Breaking changes
    - **calds**, **calpds**: Order of arguments 'Xt' and 'X' were inverted
        (required to be useable by **gridscore** etc.).

### *Version 0.1.21*

- News
    - **kdeda**: Discriminant analysis using non-parametric kernel Gaussian 
        density estimation (KDE-DA).
    - **plskdeda**: PLS-KDE-DA 

- Modifications
    - Code improvement.

- Breaking changes
    - **lda**, **qda**: returned output 'ds' renamed to 'dens'.


### *Version 0.1.20*

- News
    - **dmkern**: Gaussian kernel density estimation (KDE)..
    - **out**: Return if elements of a vector are strictly outside of a given range.
    - **pval**: Compute p-value(s) for a distribution, an ECDF or a vector.

- Modifications
    - **plotxy**: accept a matrix (n, 2) as input.
    - Code improvement.

- Breaking changes
    - **dens**: removed and replaced by **dmkern**.

### *Version 0.1.19*

- News

- Modifications
    - Dependance to unused package **HypothesisTests.jl** was removed. 
    - Code cleaning.


### *Version 0.1.18*

- News
    - **confusion** Confusion matrix.
    - **plotconf** Plot confusion matrix.

- Modifications
    - **cplsravg**: new argument 'typda'. 
    - Code cleaning.


### *Version 0.1.17*

- News 
    - **lwmlrda**: k-Nearest-Neighbours locally weighted MLR-based discrimination (kNN-LWMLR-DA).
    - **lwmlrda_s**: kNN-LWMLR-DA after preliminary (linear or non-linear) dimension reduction.
    - **lwplsrda_s** kNN-LWPLSR-DA after preliminary (linear or non-linear) dimension reduction.

- Modifications
    - **lwmlr_s**: Add arguments 'psamp' and 'samp' for large nb. observations. 
    - Code cleaning.

- Breaking changes
    - **lwplsr_s**: Arguments and pipeline changed to be consistent 
        with **lwmlr_s**. 
    - **sampclas**: remamed to **sampcla**.

### *Version 0.1.16* 

- News
    - **dkplsrda** Discrimination based on direct kernel partial least 
        squares regression (DKPLSR-DA)
    - **treer_dt** Regression tree (CART) with DecisionTree.jl
    - **rfr_dt** Random forest regression with DecisionTree.jl
    - **treeda_dt** Discrimination tree (CART) with DecisionTree.jl
    - **rfda_dt** Random forest discrimination with DecisionTree.jl

- Modifications
    - **selwold** :  add argument 'step'- 
    - Code cleaning.

- Breaking changes
    - **Warning**: Difficult breaking bugs appeared in C++ dll from Julia v1.8.4 
        (still present in v1.9-betas) that removed the possibility to use packages 
        LIBSVM.jl and XGBoost.jl under Windows. For this reason, Jchemo.jl stopped to use 
        these two packages. All the related functions (SVM, RF and XGBoost models) 
        were removed. For CART models (trees), they were replaced by new functions 
        using package DecisionTree.jl.  

### *Version 0.1.15* 

- News
    - **cosm** Cosinus between the columns of a matrix
    - **cosv** Cosinus between two vectors
    - **lwmlr**: k-Nearest-neighbours locally weighted multiple linear regression (kNN-LWMLR)
    - **lwmlr_s**: kNN-LWMLR after preliminary (linear or non-linear) dimension reduction
    - **pmod** Short-cut for function 'Base.parentmodule'
    - **tabdupl** Tabulate duplicated values in a vector

- Modifications
    - Improvement of **vi_baggr**
    - Code cleaning.

### *Version 0.1.14* 
- News
    - **isel**: Interval variable selection.
    - **mlev**: Return the sorted levels of a dataset.
    - **pcasph**: Spherical PCA.
    - **tabdf**: Compute the nb. occurences of groups in categorical variables of 
        a dataset.
    - **vip**: Variable importance by permutation.

- Modifications
    - **plotgrid**: add of argument 'leg'. 
    - **plotxy**: add of arguments 'circle' and 'zeros'. 
    - Code cleaning.

- Breaking changes
    - **aggstat** has changed (arguments).
    - **baggr_vi** renamed to **vi_baggr**
    - **baggr_oob** renamed to **oob_baggr**
    - **gridcv** and **gridcv_mb**: in output 'res_rep', colum 'rept' replaced
         by column 'repl'.
    - **iplsr** was removed and replaced by the more generic function **isel**.
    - **mtest**: Outputs 'idtrain' and 'idtest' renamed to 'train' and 'test'.
    - **rd**: argument 'corr' chanfed to 'typ'.
    - **tabn** was removed.
    - **vimp_xgb** renamed to **vi_xgb**
    - **vip**: outputs have been improved.

### *Version 0.1.13* 
- News
    - **mtest** Select indexes defining training and test sets for each column 
        of a dataframe.

- Modifications
    - Code cleaning.

### *Version 0.1.12* 
- Modifications
    - All tree functions: Internal changes to adapt to modifications in XGBoost.jl library.

### *Version 0.1.11* 
- News
    - **vip** Variable importance on PLS projections (VIP).

### *Version 0.1.10* 
- News
    - **mbwcov**: Multiblock weighted covariate analysis regression (MBWCov) (Mangana et al. 2021).

- Modifications
    - Code cleaning.

- Breaking changes
    - **mbmang** renamed to **mbunif** (Unified multiblock analysis).
    - **ramang** renamed to **rrr** (Reduced rank regression).

### *Version 0.1.9* 
- News
    - **rasvd**: Redundancy analysis - PCA on instrumental variables (PCAIV).
    - **ramang** Redundancy analysis regression = Reduced rank regression (RRR)
    - **mbplswest** MBPLSR - Nipals algorithm (Westerhuis et al. 1998) 

- Breaking changes
    - All the functions **..._avg** and **..._stack** renamed 
        to **...avg** and **...stack** (e.g. **plsr_avg** to 
        **plsravg**).
    - **caltransf_ds** and **caltransf_pds** remaned
        to **calds** and **calpds**.
    - **fnorm** renamed to **frob**.

### *Version 0.1.8* 
- News
    - **plswold**: PLSR Wold Nipals algorithm.
    - **ccawold**: CCA Wold Nipals algorithm.
    - **mbmang**: Unified multiblock data analysis of Mangana et al. 2019.

- Modifications
    - **mlrpinv_n** renamed to **mlrpinvn**
    - **pls** renamed to **plscan**.
    - **pls_svd** renamed to **plstuck**.
    - **rcca** renamed to **cca** (and argument 'alpha" to 'tau').
    - **rpmat_gauss** and **rpmat_li** renamed to **rpmatgauss** and **rpmatli** 
    - Output 'Tbl' added in **comdim** and **mbpca**.
    - Code cleaning.

### *Version 0.1.7* 
- CairoMakie.jl was removed from the dependances, and replaced by Makie.jl.
To display the plots, the user has to install and load one of
the Makie's backend (e.g. CairoMakie).

- News
    - **rcca**: Canonical correlation analysis. (RCCA).
    - **pls**: Canonical partial least squares regression (Symmetric PLS).
    - **pls_svd**: Tucker's inter-battery method of factor analysis (PLS-SVD).
    - **colnorm2** was removed, replaced by **colnorm**: 
    Norm of each column of a dataset.
    - **fnorm**: Frobenius norm of a matrix.
    - **norm2** was removed, replaced by **normw**: 
    Weighted norm of a vector.

- Modifications
    - Major changes in multiblock functions:
        - Renamed functions:
            - **mbpca_cons** ==> **mbpca**
            - **mbpca_comdim_s** ==> **comdim**
            - **mbplsr_rosa** ==> **rosaplsr**
            - **mbplsr_so** ==> **soplsr**
        - Argument 'X_bl' renamed to 'Xbl'
    - Variable 'pc' in summary outputs of PCA and KPCA functions renamed to 'lv'. 
    - Modification of all the tree functions to adapt to the new version of
    XGBoost.jl (>= 2.02) (https://juliahub.com/ui/Packages/XGBoost/rSeEh/2.0.2). 
    The new Jchemo functions does not work anymore with XGBoost.jl 1.5.2.    
    - Code cleaning.

### *Version 0.1.6*
- Package Jchemo.jl has been registered.

- Modifications
    - Code cleaning.

### *Version 0.1.5*
- Modifications
    - Code cleaning.

### *Version 0.1.4*
- News
    - **head**: Display the first rows of a dataset.

- Modifications
    - Remove of side-effects in some functions of multi-bloc analyses.

### *Version 0.1.3*
- Modifications
    - **detrend**: argument 'degree' renamed to 'pol'.
    - Code cleaning.

### *Version 0.1.2*
- Modifications
    - **detrend**: new argument 'degree'
    - **gridcvlv**: correction of a bug (typing error) inserted 
        in the last version.

### *Version 0.1.1*
- Modifications
    - **blockscal**: bug corrected in arguments.
    - Use of *multi-threading* (package Threads)
        in functions **locw** and **locwlv**, used in local models.

### *Version 0.1.0*
- Modifications
    - Argument 'scal' (X and/or Y column-scaling) added to various functions.
    - **blockscal**: names of arguments have changed.
    - **plotgrid**: argument 'indx' modified.

### *Version 0.0.26*
- News
    - **cscale**: Center and scale each column of a matrix.
    
- Modifications
    - Argument 'scal' (X and/or Y column-scaling) added to various functions.
        Work in progress. The argument will be available for all the concerned fonctions.
    - Output 'explvar' replaced by 'explvarx' in all the concerned functions.
    - **rd**: New argument 'corr'.

### *Version 0.0.25*
- News 
    - **rd**: Redundancy coefficients between two matrices.

- Modifications
    - **summary** for Plsr objects. See the example in ?plskern.

### *Version 0.0.24*
- Modifications
    - **selwold**: Argument "plot" renamed "graph" and bug fixed in plotting.

### *Version 0.0.23*
- News 
    - **occknndis**: One-class classification using "global" k-nearest neighbors distances.
    - **occlknndis**: One-class classification using "local" k-nearest neighbors distances.

- Modifications
    - **occsd**, **occod**, **occsdod**, **occstah**: The methods to compute the cutoff have changed.

### *Version 0.0.22*
- News 
    - **colmad**: Median absolute deviation (MAD) of each column of a matrix.
    - **occsdod**: One-class classification using a compromise between PCA/PLS score (SD) and orthogonal (OD) distances.
    - **replacedict**: Replace the elements of a vector by levels defined in a dictionary.
    - **stah**: Stahel-Donoho outlierness measure.

- Modifications
    - **dens**: outputs have been modified.
    - **odis** and **scordis** have been rename to **occsd** and **occod**, and modified.
    - **plotxy**: new argument "bisect".

### *Version 0.0.21* 
- News 
    - **dens**: Univariate kernel density estimation.

- Modifications 
    - All the datasets (examples) have been moved to package JchemoData
         (https://github.com/mlesnoff/JchemoData.jl)
    - **plotsp**: Argument 'nsamp' added.
    - **datasets**: removed and transferred to JchemoData.jl


### *Version 0.0.20* 
- News 
    - **covselr**: Covsel regression (Covsel+Mlr).

- Modifications 
    - **covsel**, **mlrvec**: Arguments changed.


### *Version 0.0.19* 
- News 
    - **selwold** : Wold's criterion to select dimensionality in LV (e.g. PLSR) models.
    - **plotxy** : Scatter plot (x, y) data.

- Modifications 
    - **plotscore**: Renamed to **plotgrid**.    

### *Version 0.0.18* 
- News 
    - **plotgrid** : Plot error or performance rates of model predictions.
 
- Modifications 
    - **plotsp**: argument 'resolution' was added.

### *Version 0.0.17* 
- News 
    - **replacebylev2** : Replace the elements of an index-vector by levels.

- Modifications 
    - **aggstat** : Sorting order for dataframes.
    - **checkdupl** : bug corrected.
    - **matB**, **matW** : when requested, update of covm to cov, and aggstat output.
    - **plotsp** : faster.
    - **transfer_ds** : renamed to **caltransf_ds**.
    - **transfer_pds** : renamed to **caltransf_pds**.
    - **recodcat2num** : renamed to **recodcat2int**
    - **segmts** : A seed (MersenneTwister) can be set for the random samplings.
    - Examples added in the helps of every functions.
    - Discrimination functions: major updates.

### *Version 0.0.16*
- News 
    - **transfer_ds** : Calibration transfert with direct standardization (DS).
    - **transfer_pds** : Calibration transfert with piecewise direct standardization (PDS).

- Modifications
    - **mlr** functions : Argument 'noint' added.
    - **plsr_avg_cv** : Bug corrected.

### *Version 0.0.15*
- News 
    - **plsr_stack** : Stacking PLSR models

- Modifications
    - **aicplsr** : BIC criterion added
    - **fweight**
    - **plsr_avg** : Stacking was added
    - **plsr_avg_aic**
    - **plsr_avg_cv**
    - **lwplsr_avg**

### *Version 0.0.14*
- News 
    - **lwplsr_s** 

### *Version 0.0.13*
- News 
    - **fweight** 
    - **rowmean**, **rowstd**

- Modifications
    - **aicplsr**
    - **lwplsr_avg**
    - **plsr_avg**
    - **snv**
    - **wshenk**

### *Version 0.0.12*
- News 
    - **nco**, **nro**

- Modifications
    - **mpars** renamed to **mpar**
    - All functions terminating with "..._agg" renamed to "..._avg".


### *Version 0.0.11*
- News 
    - **blockscal_mfa**
    - **datasets**
    - **mbpca_cons**
    - **lg**
    - **ssq**

- Modifications
    - All the functions terminating with a "s" have been renamed without "s"
    (e.g. **colmeans** was renamed to **colmean**)

### *Version 0.0.10*
- News 
    - **colsum**
    - **mbpca_comdim_s**
    - **rowsum**

- Modifications
    - **nipals**
    - **mse**
    - **mbpls**

### *Version 0.0.9*
- News 
    - **blockscal_frob**, **blockscal_ncol**, **blockscal_sd**
    - **colnorm2**
    - **corm**, **covm**
    - **nipals**
    - **norm2**

- Modifications
    - **blockscal** 
    - **matcov** renamed to **covm** and extended

- Removed
    - **mbplsr_mid_avg**
    - **mbplsr_mid**
    - **mbplsr_mid_seq**

### *Version 0.0.8*
- News functions
    - **gridcv_mb**
    - **gridcvlv_mb**
    - **mbplsr_avg**
    - **mbplsr_mid**
    - **mbplsr_mid_seq**
- Modifications 
    - **rosaplsr** renamed to **mbplsr_rosa**
    - **soplsr** renamed to **mbplsr_soplsr**

### *Version 0.0.7*
- News functions
    - **mbplsr**
    - **soplsr**

### *Version 0.0.6*
- News functions
    - **colstd**
    - **plsrosa**
    - **plssimp**
    - **rosaplsr**
    - **rv**
    - **rmrows**, **rmcols**: renamed to **rmrow**, **rmcol**
- Modifications 
    - **interpl**, **interpl_mon**: changes in arguments
    - **plotsp**: changes in outputs
    - **aggstat** (::AbstractMatrix): changes in arguments and outputs
 
### *Version 0.0.5*
- News functions
    - **blockscal**
    - **pcr**
    - **rp**
    - **rpmatgauss**
    - **rpmat_li**
   
### *Version 0.0.4*
- News functions
    - **iplsr**

- Modification of **covsel**

### *Version 0.0.3*
- News functions
    - **interpl**
    - **checkdupl**, **checkmiss**

### *Version 0.0.2*
- News functions
    - **covsel**
    - **interpl** has been replaced by **interpl_mon**
- Change in output of **vi_xgb**

### *Version 0.0.1*

First version of the package
