## NEWS for package **Jchemo**

### *Version 0.0-16*

- New 

   - **transfer_ds** : Calibration transfert with direct standardization (DS).
   - **transfer_pds** : Calibration transfert with piecewise direct standardization (PDS).

- Modified

    - **mlr** functions : Argument 'noint' added.
    - **plsr_avg_cv** : Bug corrected.


### *Version 0.0-15*

- New 
    - **plsr_stack** : Stacking PLSR models

- Modified

    - **aicplsr** : BIC criterion added
    - **fweight**
    - **plsr_avg** : Stacking was added
    - **plsr_avg_aic**
    - **plsr_avg_cv**
    - **lwplsr_avg**

### *Version 0.0-14*
- New 
    - **lwplsr_s** 

### *Version 0.0-13*
- New 
    - **fweight** 
    - **rowmean**, **rowstd**

- Modified
    - **aicplsr**
    - **lwplsr_avg**
    - **plsr_avg**
    - **snv**
    - **wshenk**

### *Version 0.0-12*
- New 
    - **nco**, **nro**

- Modified
    - **mpars** renamed to **mpar**
    - All functions terminating with "..._agg" renamed to "..._avg".


### *Version 0.0-11*
- New 
    - **blockscal_mfa**
    - **datasets**
    - **mbpca_cons**
    - **lg**
    - **ssq**

- Modified
    - All the functions terminating with a "s" have been renamed without "s"
    (e.g. **colmeans** was renamed to **colmean**)

### *Version 0.0-10*
- New 
    - **colsum**
    - **mbpca_comdim_s**
    - **rowsum**

- Modified
    - **nipals**
    - **mse**
    - **mbpls**

### *Version 0.0-9*
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

### *Version 0.0-8*
- New functions
    - **gridcv_mb**
    - **gridcvlv_mb**
    - **mbplsr_avg**
    - **mbplsr_mid**
    - **mbplsr_mid_seq**
- Modifications 
    - **rosaplsr** renamed to **mbplsr_rosa**
    - **soplsr** renamed to **mbplsr_so**

### *Version 0.0-7*
- New functions
    - **mbplsr**
    - **soplsr**

### *Version 0.0-6*
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
 
### *Version 0.0-5*
- New functions
    - **blockscal**
    - **pcr**
    - **rp**
    - **rpmat_gauss**
    - **rpmat_li**
   
### *Version 0.0-4*
- New functions
    - **iplsr**

- Modification of **covsel**

### *Version 0.0-3*
- New functions
    - **interpl**
    - **checkdupl**, **checkmiss**

### *Version 0.0-2*
- New functions
    - **covsel**
    - **interpl** has been replaced by **interpl_mon**
- Change in output of **vimp_xgb**

### *Version 0.0-1*

First version of the package
