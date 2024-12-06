# News

## *Version 0.6.3*

- Breaking
    - **dummy**: To improve time computations, the output 'Y' is now a BitMatrix.
    - **iqr**: renamed to **iqrv**.
    - **mad**: renamed to **madv**.
    - **normw**: renamed to **normv**.
    - **splskern**: renamed to **splsr**.

- New
    - **sumv**, **meanv**, **stdv**, **varv**: Vector operations.

- Modifications
    - Code cleaning.  

## *Version 0.6.2*

- Modifications
    - Correction of a bug in function **tab**  introduced in 0.6.1
    (unfortunate changes in the mutiple dispatch, that impacted DA functions). 
    - Code cleaning.  

## *Version 0.6.1*

- New
    - **convertdf** Convert the columns of a dataframe to given types.
    - **parsemiss** Parsing a string vector allowing missing data.
    - **recod_miss** Declare data as missing in a dataset.

- Breaking
    - **miss**: renamed to **findmiss**.
    - **tabdf**: renamed to **tab**.
    - **aggstat**, **tabdf**: arguments 'groups' and 'vars' renamed to 'vargroup' and 'vary'.

- Modifications
    - Code cleaning.    
    
## *Version 0.6.0*

**Warning:** Major breaking changes.
The function **model** of the embedded syntax has been removed. The reason is: this function was frequently entering in conflict with the variable-name 'model' used 
in many scripts built from other machine learning packages (Flux.jl, LearnAPI.jl, etc.).
Sorry for any inconvenience.

The previous syntax (<= 0.5.3):

*mod = model(plskern; nlv = 10)*

*fit!(mod, X, Y)*

is now written (see README):

*model = plskern(nlv = 10)*  # 'model' is now a variable (object) whose name can be changed 

*fit!(model, X, Y)*

or equivalently

*nlv = 10*

*model = plskern(; nlv)*

*fit!(model, X, Y)*

- New
    - **umap**: new argument 'psamp'.

- Breaking
    - **sampdf**: argument 'msamp' renamed to 'meth'.
    - Output 'fmpls' renamed to 'fmemb' in PLSDA functions.
    - Functions **dtlo**, **dtpol**, **dtasls**, **dtarpls*, **dtairpls** 
        renamed to **detrend_lo**, **detrend_pol**, **detrend_asls**, **detrend_arpls**, 
        **detrend_airpls**.
    - **dmnorm**: one of the methods has been modified.
    - Argument and object 'fun' has been renamed 'algo' everywhere.
    - Object 'fm' has been renamed 'fitm' (fitted model) everywhere.

- Modifications
    - Code cleaning.


## *Version 0.5.3*

- New
    - **loessr** Compute a locally weighted regression model (LOESS)
        (with package Loess.jl).
    - **dtlo** Baseline correction of each row of X-data by LOESS regression.

- Breaking
    - **detrend**, **asls**, **airpls**, **arpls**: renamed to 
        **dtpol**, **dtasls**, **dtairpls**, **dtarpls**.

- Modifications
    - Code cleaning.

## *Version 0.5.2*

- Modifications
    - **airpls**: Correction of the termination criterion.

## *Version 0.5.1*

- New
    - Baseline correction of each row of X-data by
        - **asls**  asymmetric least squares algorithm (ASLS).
        - **airpls** adaptive iteratively reweighted penalized least squares algorithm (AIRPLS).
        - **arpls** asymmetrically reweighted penalized least squares smoothing (ARPLS).

- Breaking
    - Function **defpar** renamed to **default**.

- Modifications
    - Code cleaning.

## *Version 0.5.0*

- New
    - The management of the arguments of the functions has been deeply modified. 
    The unique container 'Par' has been removed, and replaced by independent 
    containers between functions or groups of functions (eg. 'ParPca').
    - **defpar** Display the keyword arguments (with their default values) of a function.
    - **umap**: UMAP: Uniform manifold approximation and projection for 
    dimension reduction (uses package **UMAP.jl** added as dependence).

- Breaking
    - Argument 'nlvdis' removed from **knnr** and **knnda** (there is no possible dimension 
        reduction anymore).
    - Renaming of functions:  
        - **findindex** to **recod_catbyind**.
        - **recodcat2int** to **recod_catbyint**
        - **recodnum2int** to **recod_numbyint**
        - **replacebylev** to **recod_catbylev**
        - **replacebylev2** to **recod_indbylev**
        - **replacedict** to **recod_catbydict**
        - **treer_dt**, **treeda_dt**, **rfr_dt**, **rfda_dt** to **treer**, **treeda**,
            **rfr**, **rfda**.   
    - **rp** : argument 'mrp' renamed to 'meth'.
    - **rpmatli**: argument 's_li' renamed to 's'.
    - **spca**, **splskern** : argument 'msparse' renamed to 'meth'.

- Modifications
    - Code cleaning.

## *Version 0.4.3*

- New
    - **outeucl**: Outlierness from Euclidean distances to center.
    - **talworth**: Weight function.
    - **pcapp**, **pcapp!**: Robust PCA by projection pursuit. 
    - **pcaout**, **pcaout!**: Robust PCA using outlierness.
    - **plsrout**, **plsrout!**: Robust PLSR using outlierness.

- Breaking
    - **stah**: Renamed to **outstah** and improved. 

- Modifications
    - Code cleaning.

## *Version 0.4.2*

- New 
    - **sampwsp**: Build training vs. test sets by WSP sampling.

- Modifications
    - Code cleaning.

## *Version 0.4.1*

- Breaking
    - **savgk**: order of arguments have changed. 

- Modifications
    - Code cleaning.

## *Version 0.4.0*

**Warning:** Major breaking changes with versions 0.3. The embedded syntax 
has changed, with the use of the new function **model**:
    - 'model = plskern(; nlv = 15)' is now writen as 'model = mod_(plskern; nlv = 15)'. 

- Modifications
    - Code cleaning. The examples in th the function help pages have 
    been corrected from typing errors. 

## *Version 0.3.7*

- New 
    - **aggsum**: Compute sub-total sums by class for a categorical variable.
    - **findindex**: Replace a vector containg levels by the indexes of a set of levels.
    - **getknn**: Add of angular and correlation distances.
    - **kplslda**, **kplsqda**, **kplskdeda**: Kernel PLS-LDA, PLS-QDA, PLS-KDEDA.
    - **dkplslda**, **dkplsqda**, **dkplskdeda**: Direct kernel PLS-LDA, PLS-QDA, PLS-KDEDA.
    - **mbplslda**, **mbplsqda**, **mbplskdeda**: Multiblock PLS-LDA, PLS-QDA, PLS-KDEDA.
    - **merrp**: Mean intra-class classification error rate.
    - **mweightcla**: Compute observation weights for a categorical variable, 
        given specified sub-total weights for the classes.
    - **rownorm**: Row-wise norms.
    - **snorm**: Row-wise norming of X-data.

- Breaking
    - **confusion**: renamed to **conf**, and output 'accuracy' renamed to 'accpct' 
        and modified. 

- Modifications
    - FDA and predictive DA functions: improvements to better take into account 
        for unbalanced classes.
    - Code cleaning. 

## *Version 0.3.6*

- Modifications
    - Code improvement. 

## *Version 0.3.5*

- Modifications
    - Code improvement. 

## *Version 0.3.4*

- New
    - **colmed**: Column-wise medians.

- Modifications
    - Code improvement. 


## *Version 0.3.3*

- Modifications
    - Code improvement. 
    - **calds**, **calpds**: Function 'fit!' was added.
- Breaking
    - **rmgap**: works now with a function 'transf'. 
        Function **rmgap!** was removed.

## *Version 0.3.2*

- Modifications
    - Code improvement. 

- Breaking
    - **hconcat** renamed to **mbconcat**
    - **recodnum2cla** renamed to **recodnum2int**
    - **viperm!** renamed to **viperm**

## *Version 0.3.1*

- New 
    - **hconcat** Transformer contatenating multi-block X-data.
    - **blockscal** Scaling of multiblock X-data. Replace functions
        'fblockscal_...".
    - **mbplsrda** Discrimination based on multiblock partial 
        least squares regression (MBPLSR-DA).

- Modifications
    - Code improvement. 
    - **mbplsr**, **soplsr** : correction when 'scal' = true.

## *Version 0.3.0*

**Warning:** Major breaking changes.
Package Jchemo has been deeply restructured, to
enable an "embedded" syntax and facilitate pipelines 
building. The previous "direct" syntax is still allowed (although arugments have changed for some functions, see the help pages) but is not favor anymore. Users who prefer the previous syntax will have to keep working with versions < 0.3.0. Sorry for any inconveniance. 

Some typing errors may have been introduced due to the 
restructuration. They will be corrected in versions > 0.3.0.

Some specific modified points are:

- In the arguments, all String types have been replaced by Type Symbol (e.g. "unif" is replaced by :unif)
- The 'weights' (row weighting in some functions, such as pcasvd etc.) argument must now be of type 'Weight', built from function 'mweight'. 
- Sampling functions 'samp...' have changed. 
- Function 'mtest' renamed to 'sampdf'.
- Functions center, scale, cscale, blockscale renamed to fcenter, fscale, etc. Alternatively, new transformers center, scale and cscale have been created.
- Functions isel and viperm remated to isel! and viperm!, and syntax changed.
- Syntax of tuning functions gridscore and gridcv
    has changed, and the functions are now genereic 
    (no need anymore to call specific functions ...lv 
    and ...lb).
- Removed function (temporary or not):
    - baggr and its utilities
    - cplsravg
    - gridcv_mb, gridcvlv_mb
    - interpl_mon
    - lwmlr_s, lwplsr_s, lwmlrda_s, lwlsrda_s 
        (since can be built from pipelines)
    - lwplsrdaavg, lwplsldaavg, lwplsqdaavg
    - mavg_runmean
    - mbunif, mbwcov
    - nsc, nscrda, nscda
    - occknndis, occlknndis
    - plsrdaavg, plsldaavg, plsqdaavg

## *Version 0.2.4*

- Modifications
    - Code improvement.

## *Version 0.2.3*

- New
    - **samprand**: Build training vs. test sets by random sampling.
    - **svmr**, **svmda**: SVM (wrappers to LIBSVM.jl) were reset in 
        the package.

- Modifications
    - **baggr** has been parallelized (multi-threading).
    - Code improvement.

- Breaking changes
    - **sampks**, **sampdp**, **samprand**, **sampsys**, **sampcla**: order of arguments have changed.
    - **mtest**: Arguments have changed, and function has been improved. 
    - **plotxy**: The number of methods have been reduced (for simplification). 
    The aggregated syntax (matrice with two columns as input {x,y}) is not 
    allowed anymore. 

## *Version 0.2.2*

- New
    - **@head**: macro for function 'head'.

- Modifications
    - Code improvement.

## *Version 0.2.1*

- New
    - **splsrda**: Sparse PLSR-DA.
    - **splslda**: Sparse PLS-LDA.
    - **splsqda**: Sparse PLS-QDA.
    - **splskdeda**: Sparse PLS-KDE-DA.

- Modifications
    - Code improvement.
    
## *Version 0.2.0*

- New
    - **splskern**: Sparse PLSR.

- Modifications
    - Code improvement.

## *Version 0.1.24*

- New
    - **colmeanskip**, **colstdskip**, **colsumskip**, **colvarskip**: Column-wise operations allowing missing data.
    - **nsc**: Nearest shrunken centroids (NSC).
    - **nscda**: Discrimination by nearest shrunken centroids.
    - **pcanipals**: PCA by NIPALS algorithm.
    - **pcanipalsmiss**: PCA by NIPALS algorithm allowing missing data.
    - **plist** Print each element of a list.
    - **rowmeanskip**, **rowstdskip**, **rowsumskip**, **rowvarskip**: Row-wise operations allowing missing data.
    - **soft**: Soft thresholding.
    - **softmax**: Softmax function.
    - **spca**, **snipals**, **snipalsh**, **snipalsmix**: Sparse PCA (Shen & Hunag 2008).

- Modifications
    - Code improvement.

- Breaking changes
    - **covsel** has been removed and integrated into **covselr** that was
        extended and made faster. 

## *Version 0.1.23*

- New
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

## *Version 0.1.22*

- New
    - **difmean** : Compute a detrimental matrix (for calibration transfer) by column 
        means difference.

- Modifications
    - Code improvement.

- Breaking changes
    - **calds**, **calpds**: Order of arguments 'Xt' and 'X' were inverted
        (required to be useable by **gridscore** etc.).

## *Version 0.1.21*

- New
    - **kdeda**: Discriminant analysis using non-parametric kernel Gaussian 
        density estimation (KDE-DA).
    - **plskdeda**: PLS-KDE-DA 

- Modifications
    - Code improvement.

- Breaking changes
    - **lda**, **qda**: returned output 'ds' renamed to 'dens'.

## *Version 0.1.20*

- New
    - **dmkern**: Gaussian kernel density estimation (KDE)..
    - **out**: Return if elements of a vector are strictly outside of a given range.
    - **pval**: Compute p-value(s) for a distribution, an ECDF or a vector.

- Modifications
    - **plotxy**: accept a matrix (n, 2) as input.
    - Code improvement.

- Breaking changes
    - **dens**: removed and replaced by **dmkern**.

## *Version 0.1.19*

- New

- Modifications
    - Dependance to unused package **HypothesisTests.jl** was removed. 
    - Code cleaning.

## *Version 0.1.18*

- New
    - **confusion** Confusion matrix.
    - **plotconf** Plot confusion matrix.

- Modifications
    - **cplsravg**: new argument 'typda'. 
    - Code cleaning.

## *Version 0.1.17*

- New 
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

## *Version 0.1.16* 

- New
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

## *Version 0.1.15* 

- New
    - **cosm** Cosinus between the columns of a matrix
    - **cosv** Cosinus between two vectors
    - **lwmlr**: k-Nearest-neighbours locally weighted multiple linear regression (kNN-LWMLR)
    - **lwmlr_s**: kNN-LWMLR after preliminary (linear or non-linear) dimension reduction
    - **pmod** Short-cut for function 'Base.parentmodule'
    - **tabdupl** Tabulate duplicated values in a vector

- Modifications
    - Improvement of **vi_baggr**
    - Code cleaning.

## *Version 0.1.14* 

- New
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

## *Version 0.1.13* 

- New
    - **mtest** Select indexes defining training and test sets for each column 
        of a dataframe.

- Modifications
    - Code cleaning.

## *Version 0.1.12* 
- Modifications
    - All tree functions: Internal changes to adapt to modifications in XGBoost.jl library.

## *Version 0.1.11* 
- New
    - **vip** Variable importance on PLS projections (VIP).

## *Version 0.1.10* 

- New
    - **mbwcov**: Multiblock weighted covariate analysis regression (MBWCov) (Mangana et al. 2021).

- Modifications
    - Code cleaning.

- Breaking changes
    - **mbmang** renamed to **mbunif** (Unified multiblock analysis).
    - **ramang** renamed to **rrr** (Reduced rank regression).

## *Version 0.1.9* 

- New
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

## *Version 0.1.8* 

- New
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

## *Version 0.1.7* 

- CairoMakie.jl was removed from the dependances, and replaced by Makie.jl.
To display the plots, the user has to install and load one of
the Makie's backend (e.g. CairoMakie).

- New
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

## *Version 0.1.6*

- Package Jchemo.jl has been registered.

- Modifications
    - Code cleaning.

## *Version 0.1.5*

- Modifications
    - Code cleaning.

## *Version 0.1.4*

- New
    - **head**: Display the first rows of a dataset.

- Modifications
    - Remove of side-effects in some functions of multi-bloc analyses.

## *Version 0.1.3*

- Modifications
    - **detrend**: argument 'degree' renamed to 'pol'.
    - Code cleaning.

## *Version 0.1.2*

- Modifications
    - **detrend**: new argument 'degree'
    - **gridcvlv**: correction of a bug (typing error) inserted 
        in the last version.

## *Version 0.1.1*

- Modifications
    - **blockscal**: bug corrected in arguments.
    - Use of *multi-threading* (package Threads)
        in functions **locw** and **locwlv**, used in local models.

## *Version 0.1.0*

- Modifications
    - Argument 'scal' (X and/or Y column-scaling) added to various functions.
    - **blockscal**: names of arguments have changed.
    - **plotgrid**: argument 'indx' modified.

## *Version 0.0.26*

- New
    - **cscale**: Center and scale each column of a matrix.
    
- Modifications
    - Argument 'scal' (X and/or Y column-scaling) added to various functions.
        Work in progress. The argument will be available for all the concerned fonctions.
    - Output 'explvar' replaced by 'explvarx' in all the concerned functions.
    - **rd**: New argument 'corr'.

## *Version 0.0.25*

- New 
    - **rd**: Redundancy coefficients between two matrices.

- Modifications
    - **summary** for Plsr objects. See the example in ?plskern.

## *Version 0.0.24*

- Modifications
    - **selwold**: Argument "plot" renamed "graph" and bug fixed in plotting.

## *Version 0.0.23*

- New 
    - **occknndis**: One-class classification using "global" k-nearest neighbors distances.
    - **occlknndis**: One-class classification using "local" k-nearest neighbors distances.

- Modifications
    - **occsd**, **occod**, **occsdod**, **occstah**: The methods to compute the cutoff have changed.

## *Version 0.0.22*

- New 
    - **colmad**: Median absolute deviation (MAD) of each column of a matrix.
    - **occsdod**: One-class classification using a compromise between PCA/PLS score (SD) and orthogonal (OD) distances.
    - **replacedict**: Replace the elements of a vector by levels defined in a dictionary.
    - **stah**: Stahel-Donoho outlierness measure.

- Modifications
    - **dens**: outputs have been modified.
    - **odis** and **scordis** have been rename to **occsd** and **occod**, and modified.
    - **plotxy**: new argument "bisect".

## *Version 0.0.21* 

- New 
    - **dens**: Univariate kernel density estimation.

- Modifications 
    - All the datasets (examples) have been moved to package JchemoData
         (https://github.com/mlesnoff/JchemoData.jl)
    - **plotsp**: Argument 'nsamp' added.
    - **datasets**: removed and transferred to JchemoData.jl

## *Version 0.0.20* 

- New 
    - **covselr**: Covsel regression (Covsel+Mlr).

- Modifications 
    - **covsel**, **mlrvec**: Arguments changed.

## *Version 0.0.19* 

- New 
    - **selwold** : Wold's criterion to select dimensionality in LV (e.g. PLSR) models.
    - **plotxy** : Scatter plot (x, y) data.

- Modifications 
    - **plotscore**: Renamed to **plotgrid**.    

## *Version 0.0.18* 

- New 
    - **plotgrid** : Plot error or performance rates of model predictions.
 
- Modifications 
    - **plotsp**: argument 'size' was added.

## *Version 0.0.17* 

- New 
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

## *Version 0.0.16*

- New 
    - **transfer_ds** : Calibration transfert with direct standardization (DS).
    - **transfer_pds** : Calibration transfert with piecewise direct standardization (PDS).

- Modifications
    - **mlr** functions : Argument 'noint' added.
    - **plsr_avg_cv** : Bug corrected.

## *Version 0.0.15*

- New 
    - **plsr_stack** : Stacking PLSR models

- Modifications
    - **aicplsr** : BIC criterion added
    - **fweight**
    - **plsr_avg** : Stacking was added
    - **plsr_avg_aic**
    - **plsr_avg_cv**
    - **lwplsr_avg**

## *Version 0.0.14*

- New 
    - **lwplsr_s** 

## *Version 0.0.13*

- New 
    - **fweight** 
    - **rowmean**, **rowstd**

- Modifications
    - **aicplsr**
    - **lwplsr_avg**
    - **plsr_avg**
    - **snv**
    - **wshenk**

## *Version 0.0.12*

- New 
    - **nco**, **nro**

- Modifications
    - **mpars** renamed to **mpar**
    - All functions terminating with "..._agg" renamed to "..._avg".

## *Version 0.0.11*

- New 
    - **blockscal_mfa**
    - **datasets**
    - **mbpca_cons**
    - **lg**
    - **ssq**

- Modifications
    - All the functions terminating with a "s" have been renamed without "s"
    (e.g. **colmeans** was renamed to **colmean**)

## *Version 0.0.10*

- New 
    - **colsum**
    - **mbpca_comdim_s**
    - **rowsum**

- Modifications
    - **nipals**
    - **mse**
    - **mbpls**

## *Version 0.0.9*

- New 
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

## *Version 0.0.8*

- New functions
    - **gridcv_mb**
    - **gridcvlv_mb**
    - **mbplsr_avg**
    - **mbplsr_mid**
    - **mbplsr_mid_seq**
- Modifications 
    - **rosaplsr** renamed to **mbplsr_rosa**
    - **soplsr** renamed to **mbplsr_soplsr**

## *Version 0.0.7*

- New functions
    - **mbplsr**
    - **soplsr**

## *Version 0.0.6*

- New functions
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
 
## *Version 0.0.5*

- New functions
    - **blockscal**
    - **pcr**
    - **rp**
    - **rpmatgauss**
    - **rpmat_li**
   
## *Version 0.0.4*

- New functions
    - **iplsr**

- Modification of **covsel**

## *Version 0.0.3*

- New functions
    - **interpl**
    - **checkdupl**, **checkmiss**

## *Version 0.0.2*

- New functions
    - **covsel**
    - **interpl** has been replaced by **interpl_mon**
- Change in output of **vi_xgb**

## *Version 0.0.1*

First version of the package
