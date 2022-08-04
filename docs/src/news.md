### NEWS for package **Jchemo**

### *Version 0.0.22*
- New 
    - **colmad**: Median absolute deviation (MAD) of each column of a matrix.
    - **occsdod**: One-class classification using a compromise between PCA/PLS score (SD) and orthogonal (OD) distances.
    - **replacedict**: Replace the elements of a vector by levels defined in a dictionary.
    - **stah**: Stahel-Donoho outlierness measure.

- Modified
    - **dens**: outputs have been modified
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
    - **soplsr** renamed to **mbplsr_so**

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
