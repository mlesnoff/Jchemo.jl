module Jchemo  # Start-Module

using LinearAlgebra, SparseArrays 
using Statistics, Random
using Clustering
using DataInterpolations
using Distributions
using DataFrames
using Distances
using GLMakie
using HypothesisTests
using ImageFiltering
using Interpolations
using LIBSVM
using NearestNeighbors
using StatsBase    # sample
using XGBoost

include("utility.jl") 
include("preprocessing.jl") 
include("rmgap.jl")
include("aov1.jl")
include("plots.jl")
include("matW.jl")

include("fda.jl")
include("pcasvd.jl") ; include("pcaeigen.jl")
include("kpca.jl")
include("rp.jl")

# Regression 
include("mlr.jl")
include("rr.jl")
include("plskern.jl") ; include("plsnipals.jl") 
include("plsrosa.jl") ; include("plssimp.jl") 
include("plsr_agg.jl")
include("cglsr.jl")
include("pcr.jl")
include("krr.jl") ; include("kplsr.jl") ; include("dkplsr.jl")
include("aicplsr.jl")
include("wshenk.jl")  

# SVM
include("svmr.jl")

# Bagging
include("baggr.jl") ; include("baggr_util.jl")

# Trees
include("treer_xgb.jl")
include("treeda_xgb.jl")

include("xfit.jl")
include("scordis.jl")

# Discrimination 
include("dmnorm.jl")
include("rrda.jl")
include("lda.jl") ; include("qda.jl")
include("mlrda.jl")
include("plsrda.jl")
include("plslda.jl")
include("plsrda_agg.jl")

# SVM
include("svmda.jl")

# Local regression
include("locw.jl")
include("knnr.jl")
include("lwplsr.jl")
include("lwplsr_agg.jl")
include("cplsr_agg.jl")  # Use structure PlsrDa

# Multi-block regresssion
include("blockscal.jl")
include("angles.jl")
include("rosaplsr.jl") 

# Local discrimination
include("lwplsrda.jl") ; include("lwplslda.jl") ; include("lwplsqda.jl")
include("lwplsrda_agg.jl") ; include("lwplslda_agg.jl") ; include("lwplsqda_agg.jl")
include("knnda.jl")

# Variable selection/importance (direct methods) 
include("covsel.jl")
include("iplsr.jl")

# Validation
include("mpars.jl")
include("scores.jl")
include("gridscore.jl")
include("segm.jl") ; include("gridcv.jl")

# Sampling
include("sampling.jl")

include("distances.jl")
include("getknn.jl")
include("wdist.jl")
include("kernels.jl")

export 
    # Utilities
    aggstat,
    aov1,
    checkdupl, checkmiss,
    center, center!, 
    colmeans, colvars, colstds, 
    dummy,
    ensure_df, ensure_mat,
    findmax_cla, 
    list, 
    mweights,
    matB, matcov, matW, 
    pnames, psize,
    recodcat2num, recodnum2cla,
    replacebylev,
    rmcol, rmrow,    
    scale, scale!,
    sourcedir,
    summ,  
    tab, tabnum,
    vcol, vrow, 
   # Pre-processing
    snv, snv!, detrend, detrend!, fdif, fdif!,
    mavg, mavg!, mavg_runmean, mavg_runmean!,
    savgk, savgol, savgol!,
    interpl, interpl_mon, 
    linear_int, quadratic_int, quadratic_spline, cubic_spline,
    rmgap, rmgap!,
    eposvd,
    # Pca
    pcasvd, pcasvd!, pcaeigen, pcaeigen!, pcaeigenk, pcaeigenk!,
    kpca,
    scordis, odis,
    rpmat_gauss, rpmat_li, rp,
    # Regression
    mlr, mlr!, mlrchol, mlrchol!, mlrpinv, mlrpinv!, mlrpinv_n, mlrpinv_n!,
    mlrvec!, mlrvec,
    plskern, plskern!, plsnipals, plsnipals!, 
    plsrosa, plsrosa!, plssimp, plssimp!,
    cglsr, cglsr!,
    pcr, 
    rr, rr!, rrchol, rrchol!,   
    krr, kplsr, kplsr!, dkplsr, dkplsr!,
    plsr_agg, plsr_agg!,
    dfplsr_cg, aicplsr,
    svmr,   
    treer_xgb, rfr_xgb, xgboostr, vimp_xgb,
    baggr, baggr_vi, baggr_oob,
    # Multi-block
    mblocks, blockscal, rv,
    rosaplsr, rosaplsr!,
    # Variable selection/importance (direct methods) 
    covsel,
    iplsr,
    #
    xfit, xfit!, xresid, xresid!,
    # Local regression
    locw, locwlv,
    knnr, lwplsr, lwplsr_agg,
    cplsr_agg,
    # Discrimination
    dmnorm, dmnorm!,
    fda, fda!, fdasvd, fdasvd!,
    mlrda,
    rrda, krrda,
    lda, qda, 
    plsrda, kplsrda,
    plslda, plsqda,
    plsrda_agg, plslda_agg, plsqda_agg,
    svmda,
    treeda_xgb, rfda_xgb, xgboostda,
    # Local Discrimination
    lwplsrda, lwplslda, lwplsqda,
    lwplsrda_agg, lwplslda_agg, lwplsqda_agg,
    knnda,
    #
    transform, coef, predict,
    # Validation
    residreg, residcla, 
    ssr, msep, rmsep, bias, sep, cor2, r2, rpd, rpdr, mse, err,
    mpars,
    gridscore, gridscorelv, gridscorelb,
    segmts, segmkf,
    gridcv, gridcvlv, gridcvlb,
    # Sampling
    sampks, sampdp, sampsys, sampclas,
    # Distances
    getknn, wdist, wdist!,
    euclsq, mahsq, mahsqchol,
    krbf, kpol,
    # Graphics
    plotsp
    # Not exported since surchage:
    # - summary => Base.summary

end # End-Module




