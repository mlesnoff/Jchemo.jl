### NEWS for package **Jchemo**

### *Version 0.1.8* 
- New
    - **plswold**: Wold Nipals PLSR algorithm

- Modified
    - **pls** renamed to **pls_can**
    - **pls_svd** renamed to **pls_tuck**
    - **rcca** renamed to **cca** (and argument 'alpha" to 'tau')
    - Output 'Tbl' added in **comdim** and **mbpca**
    - Code cleaning.

### *Version 0.1.7* 
- CairoMakie.jl was removed from the dependances, and replaced by Makie.jl.
To display the plots, the user has to install and load one of
the Makie's backend (e.g. CairoMakie).

- New
    - **rcca**: Regularized canonical correlation analysis. (RCCA).
    - **pls**: Canonical partial least squares regression (Symmetric PLS).
    - **pls_svd**: Tucker's inter-battery method of factor analysis (PLS-SVD).
    - **colnorm2** was removed, replaced by **colnorm**: 
    Norm of each column of a dataset.
    - **fnorm**: Frobenius norm of a matrix.
    - **norm2** was removed, replaced by **normw**: 
    Weighted norm of a vector.

- Modified
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

- Modified
    - Code cleaning.

### *Version 0.1.5*
- Modified
    - Code cleaning.

### *Version 0.1.4*
- New
    - **head**: Display the first rows of a dataset.

- Modified
    - Remove of side-effects in some functions of multi-bloc analyses.

### *Version 0.1.3*
- Modified
    - **detrend**: argument 'degree' renamed to 'pol'.
    - Code cleaning.

### *Version 0.1.2*
- Modified
    - **detrend**: new argument 'degree'
    - **gridcvlv**: correction of a bug (typing error) inserted 
        in the last version.

### *Version 0.1.1*
- Modified
    - **blockscal**: bug corrected in arguments.
    - Use of *multi-threading* (package Threads)
        in functions **locw** and **locwlv**, used in local models.

### *Version 0.1.0*
- Modified
    - Argument 'scal' (X and/or Y column-scaling) added to various functions.
    - **blockscal**: names of arguments have changed.
    - **plotgrid**: argument 'indx' modified.

### *Version 0.0.26*
- New
    - **cscale**: Center and scale each column of a matrix.
    
- Modified
    - Argument 'scal' (X and/or Y column-scaling) added to various functions.
        Work in progress. The argument will be available for all the concerned fonctions.
    - Output 'explvar' replaced by 'explvarx' in all the concerned functions.
    - **rd**: New argument 'corr'.

### *Version 0.0.25*
- New 
    - **rd**: Redundancy coefficients between two matrices.

- Modified
    - **summary** for Plsr objects. See the example in ?plskern.

### *Version 0.0.24*
- Modified
    - **selwold**: Argument "plot" renamed "graph" and bug fixed in plotting.

### *Version 0.0.23*
- New 
    - **occknndis**: One-class classification using "global" k-nearest neighbors distances.
    - **occlknndis**: One-class classification using "local" k-nearest neighbors distances.

- Modified
    - **occsd**, **occod**, **occsdod**, **occstah**: The methods to compute the cutoff have changed.

### *Version 0.0.22*
- New 
    - **colmad**: Median absolute deviation (MAD) of each column of a matrix.
    - **occsdod**: One-class classification using a compromise between PCA/PLS score (SD) and orthogonal (OD) distances.
    - **replacedict**: Replace the elements of a vector by levels defined in a dictionary.
    - **stah**: Stahel-Donoho outlierness measure.

- Modified
    - **dens**: outputs have been modified.
    - **odis** and **scordis** have been rename to **occsd** and **occod**, and modified.
    - **plotxy**: new argument "bisect".

### *Version 0.0.21* 
- New 
    - **dens**: Univariate kernel density estimation.

- Modified 
    - All the datasets (examples) have been moved to package JchemoData
         (https://github.com/mlesnoff/JchemoData.jl)
    - **plotsp**: Argument 'nsamp' added.
    - **datasets**: removed and transferred to JchemoData.jl


### *Version 0.0.20* 
- New 
    - **covselr**: Covsel regression (Covsel+Mlr).

- Modified 
    - **covsel**, **mlrvec**: Arguments changed.


### *Version 0.0.19* 
- New 
    - **selwold** : Wold's criterion to select dimensionality in LV (e.g. PLSR) models.
    - **plotxy** : Scatter plot (x, y) data.

- Modified 
    - **plotscore**: Renamed to **plotgrid**.    

### *Version 0.0.18* 
- New 
    - **plotgrid** : Plot error or performance rates of model predictions.
 
- Modified 
    - **plotsp**: argument 'resolution' was added.

### *Version 0.0.17* 
- New 
    - **replacebylev2** : Replace the elements of an index-vector by levels.

- Modified 
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
- New 
    - **transfer_ds** : Calibration transfert with direct standardization (DS).
    - **transfer_pds** : Calibration transfert with piecewise direct standardization (PDS).

- Modified
    - **mlr** functions : Argument 'noint' added.
    - **plsr_avg_cv** : Bug corrected.

### *Version 0.0.15*
- New 
    - **plsr_stack** : Stacking PLSR models

- Modified
    - **aicplsr** : BIC criterion added
    - **fweight**
    - **plsr_avg** : Stacking was added
    - **plsr_avg_aic**
    - **plsr_avg_cv**
    - **lwplsr_avg**

### *Version 0.0.14*
- New 
    - **lwplsr_s** 

### *Version 0.0.13*
- New 
    - **fweight** 
    - **rowmean**, **rowstd**

- Modified
    - **aicplsr**
    - **lwplsr_avg**
    - **plsr_avg**
    - **snv**
    - **wshenk**

### *Version 0.0.12*
- New 
    - **nco**, **nro**

- Modified
    - **mpars** renamed to **mpar**
    - All functions terminating with "..._agg" renamed to "..._avg".


### *Version 0.0.11*
- New 
    - **blockscal_mfa**
    - **datasets**
    - **mbpca_cons**
    - **lg**
    - **ssq**

- Modified
    - All the functions terminating with a "s" have been renamed without "s"
    (e.g. **colmeans** was renamed to **colmean**)

### *Version 0.0.10*
- New 
    - **colsum**
    - **mbpca_comdim_s**
    - **rowsum**

- Modified
    - **nipals**
    - **mse**
    - **mbpls**

### *Version 0.0.9*
- New 
    - **blockscal_frob**, **blockscal_ncol**, **blockscal_sd**
    - **colnorm2**
    - **corm**, **covm**
    - **nipals**
    - **norm2**

- Modified
    - **blockscal** 
    - **matcov** renamed to **covm** and extended

- Removed
    - **mbplsr_mid_avg**
    - **mbplsr_mid**
    - **mbplsr_mid_seq**

### *Version 0.0.8*
- New functions
    - **gridcv_mb**
    - **gridcvlv_mb**
    - **mbplsr_avg**
    - **mbplsr_mid**
    - **mbplsr_mid_seq**
- Modifications 
    - **rosaplsr** renamed to **mbplsr_rosa**
    - **soplsr** renamed to **mbplsr_soplsr**

### *Version 0.0.7*
- New functions
    - **mbplsr**
    - **soplsr**

### *Version 0.0.6*
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
 
### *Version 0.0.5*
- New functions
    - **blockscal**
    - **pcr**
    - **rp**
    - **rpmat_gauss**
    - **rpmat_li**
   
### *Version 0.0.4*
- New functions
    - **iplsr**

- Modification of **covsel**

### *Version 0.0.3*
- New functions
    - **interpl**
    - **checkdupl**, **checkmiss**

### *Version 0.0.2*
- New functions
    - **covsel**
    - **interpl** has been replaced by **interpl_mon**
- Change in output of **vimp_xgb**

### *Version 0.0.1*

First version of the package
