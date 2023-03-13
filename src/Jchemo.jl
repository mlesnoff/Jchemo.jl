module Jchemo  # Start-Module

using Clustering
using DataInterpolations
using Distributions
using DataFrames
using Distances
using HypothesisTests
using ImageFiltering
using Interpolations
using LIBSVM
using LinearAlgebra
using Makie
using NearestNeighbors
using Random
using SparseArrays 
using Statistics
using StatsBase    # sample
using XGBoost

include("utility.jl") 
include("colmedspa.jl")
include("dens.jl")
include("fweight.jl") 
include("ellipse.jl")
include("matW.jl")
include("nipals.jl")
include("plotgrid.jl")
include("plotsp.jl")
include("plotxy.jl")
include("preprocessing.jl") 
include("rmgap.jl")

# Exploratory
include("fda.jl")     # Here since struct in pcasvd
include("pcasvd.jl")
include("pcaeigen.jl")
include("kpca.jl")
include("rp.jl")
include("pcasph.jl") 

# Exploratory - Multiblock 
include("angles.jl")
include("mblock.jl")
include("blockscal.jl")
include("mbpca.jl")
include("comdim.jl")
include("mbunif.jl")
include("cca.jl")
include("ccawold.jl")
include("plscan.jl")
include("plstuck.jl")
include("rasvd.jl")

# Regression 
include("aov1.jl")
include("mlr.jl")
include("rr.jl")
include("pcr.jl")
include("rrr.jl") 
include("plskern.jl") ; include("plsrosa.jl")
include("plsnipals.jl") ; include("plssimp.jl")
include("plswold.jl") 
include("plsravg.jl")
include("plsravg_aic.jl")
include("plsravg_cv.jl")
include("plsravg_unif.jl")
include("plsravg_shenk.jl")
include("plsrstack.jl")
include("cglsr.jl")
include("covselr.jl")  
include("krr.jl")
include("kplsr.jl") ; include("dkplsr.jl")
include("aicplsr.jl")
include("wshenk.jl") 
include("vip.jl") 

# Regression Multiblock
include("mbplsr.jl") 
include("mbplswest.jl")
include("mbwcov.jl")
include("rosaplsr.jl") 
include("soplsr.jl") 

# SVM
include("svmr.jl")

# Bagging
include("baggr.jl") ; include("baggr_util.jl")

# Trees
include("treer_xgb.jl")
include("treeda_xgb.jl")

include("xfit.jl")

# Discrimination 
include("dmnorm.jl")
include("rrda.jl")
include("lda.jl") ; include("qda.jl")
include("mlrda.jl")
include("occsd.jl") ; include("occod.jl") ; ; include("occsdod.jl")
include("occstah.jl") ; include("stah.jl")
include("occknndis.jl") ; include("occlknndis.jl")
include("plsrda.jl") 
include("plslda.jl") ; include("plsqda.jl")
include("plsrdaavg.jl") ; include("plsldaavg.jl") ; include("plsqdaavg.jl") 
include("krrda.jl")
include("kplsrda.jl")

# SVM
include("svmda.jl")

# Local regression
include("locw.jl")
include("locwlv.jl")
include("knnr.jl")
include("lwmlr.jl")
include("lwmlr_s.jl")
include("lwplsr.jl")
include("lwplsravg.jl")
include("lwplsr_s.jl")
include("cplsravg.jl")  # Use structure PlsrDa

# Local discrimination
include("lwplsrda.jl") ; include("lwplslda.jl") ; include("lwplsqda.jl")
include("lwplsrdaavg.jl") ; include("lwplsldaavg.jl") ; include("lwplsqdaavg.jl")
include("knnda.jl")

# Variable selection/importance (direct methods) 
include("covsel.jl")
include("isel.jl")
include("viperm.jl")

# Validation
include("mpar.jl")
include("scores.jl")
include("gridscore.jl")
include("segm.jl")
include("gridcv.jl")
include("gridcv_mb.jl")
include("selwold.jl")

# Transfer
include("calds.jl")
include("calpds.jl")
include("eposvd.jl")

# Sampling
include("mtest.jl")
include("sampling.jl")

include("distances.jl")
include("getknn.jl")
include("wdist.jl")
include("kernels.jl")

export 
    # Utilities
    aggstat,
    checkdupl, checkmiss,
    center, center!, 
    colmad, colmean, colnorm, colstd, colsum, colvar,
    corm, covm,
    cscale, cscale!, 
    dens,
    dummy,
    ensure_df, ensure_mat,
    findmax_cla, 
    frob,
    fweight,
    head,
    list, 
    matB, matW, 
    mblock,
    mlev,
    mweight, mweight!,
    nco,
    normw,  
    nro,
    pnames, psize,
    recodcat2int, replacebylev2, recodnum2cla,
    replacebylev, replacedict, 
    rmcol, rmrow, 
    rowmean, rowstd, rowsum,   
    scale, scale!,
    sourcedir,
    ssq,
    summ,
    tab, tabdf, tabdupl,
    vcatdf,
    vcol, vrow, 
   # Pre-processing
    detrend, detrend!, 
    fdif, fdif!,
    interpl, interpl_mon, 
    linear_int, quadratic_int, quadratic_spline, cubic_spline,
    mavg, mavg!, mavg_runmean, mavg_runmean!,
    rmgap, rmgap!,
    savgk, savgol, savgol!,
    snv, snv!, 
    # Transfer
    calds, calpds,
    eposvd,
    # Exploratory
    kpca,
    nipals,
    pcasvd, pcasvd!, pcaeigen, pcaeigen!, pcaeigenk, pcaeigenk!,
    rpmatgauss, rpmatli, rp, rp!,
    pcasph, pcasph!,
    # Exploratory Multiblock
    blockscal, blockscal_frob, blockscal_mfa,
    blockscal_ncol, blockscal_sd,
    rv, lg, rd, 
    mbpca, mbpca!,
    comdim, comdim!,
    mbunif, mbunif!,  
    cca, cca!,
    ccawold, ccawold!,
    plscan, plscan!,
    plstuck, plstuck!,
    rasvd, rasvd!,
    # Regression
    aov1,
    mlr, mlr!, mlrchol, mlrchol!, 
    mlrpinv, mlrpinv!, mlrpinvn, mlrpinvn!,
    mlrvec, mlrvec!,
    plskern, plskern!, 
    plsnipals, plsnipals!, 
    plsrosa, plsrosa!, 
    plssimp, plssimp!,
    plswold, plswold!,
    cglsr, cglsr!,
    pcr,
    covselr,
    rr, rr!, rrchol, rrchol!,
    rrr, rrr!,   
    krr, krr!, kplsr, kplsr!, dkplsr, dkplsr!,
    plsravg, plsravg!,
    dfplsr_cg, aicplsr,
    wshenk,
    svmr,   
    treer_xgb, rfr_xgb, xgboostr, vi_xgb,
    baggr, vi_baggr, oob_baggr,
    # Rgression Multi-block
    mbplsr, mbplsr!,
    mbplswest, mbplswest!,
    mbwcov!, mbwcov,
    rosaplsr, rosaplsr!,
    soplsr,
    # Variable selection/importance (direct methods) 
    covsel,
    isel,
    vip, viperm,
    #
    xfit, xfit!, xresid, xresid!,
    # Local regression
    locw, locwlv,
    knnr,
    lwmlr, lwmlr_s,
    lwplsr, lwplsravg, lwplsr_s,  
    cplsravg,
    # Discrimination
    dmnorm, dmnorm!,
    fda, fda!, fdasvd, fdasvd!,
    mlrda,
    rrda, krrda,
    lda, qda, 
    occsd, occod, occsdod,
    occstah, stah,
    occknndis, occlknndis,
    plsrda, kplsrda,
    plslda, plsqda,
    plsrdaavg, plsldaavg, plsqdaavg,
    svmda,
    treeda_xgb, rfda_xgb, xgboostda,
    # Local Discrimination
    lwplsrda, lwplslda, lwplsqda,
    lwplsrdaavg, lwplsldaavg, lwplsqdaavg,
    knnda,
    #
    transform, coef, predict,
    # Validation
    residreg, residcla, 
    ssr, msep, rmsep, rmsepstand, bias, sep, cor2, r2, rpd, rpdr, mse, err,
    mpar,
    gridscore, gridscorelv, gridscorelb,
    segmts, segmkf,
    gridcv, gridcvlv, gridcvlb, 
    gridcv_mb, gridcvlv_mb,
    selwold,
    # Sampling
    mtest,
    sampks, sampdp, sampsys, sampclas,
    # Distances
    getknn, wdist, wdist!,
    euclsq, mahsq, mahsqchol,
    krbf, kpol,
    # Graphics
    plotgrid,
    plotsp,
    plotxy
    # Not exported since surcharge:
    # - summary => Base.summary

end # End-Module




