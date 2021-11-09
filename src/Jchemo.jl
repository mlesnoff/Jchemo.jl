module Jchemo  # Start-Module

using LinearAlgebra, Statistics, Random
using GLMakie
using Distributions
using StatsBase    # sample
using HypothesisTests
using DataFrames
using ImageFiltering
using Interpolations
using Distances
using LIBSVM
using NearestNeighbors
using XGBoost
using Clustering

include("utility.jl") 
include("preprocessing.jl") 
include("rmgap.jl")
include("aov1.jl")
include("plots.jl")
include("matW.jl")

include("fda.jl")
include("pcasvd.jl") ; include("pcaeigen.jl")
include("kpca.jl")

# Regression 
include("mlr.jl")
include("rr.jl")
include("plskern.jl") ; include("plsnipals.jl") 
include("cglsr.jl") ; include("aicplsr.jl") ; include("wshenk.jl")  
include("krr.jl") ; include("kplsr.jl") ; include("dkplsr.jl")
include("plsr_agg.jl")

# SVM
include("svmr.jl")

# Bagging
include("baggr.jl") ; include("baggr_util.jl")

# Trees
include("treer_xgb.jl")

include("xfit.jl")
include("scordis.jl")

# Discrimination 
include("dmnorm.jl")
include("lda.jl") ; include("qda.jl")
include("mlrda.jl")
include("plsrda.jl")
include("rrda.jl")
include("plslda.jl")

# Local regression
include("locw.jl")
include("knnr.jl")
include("lwplsr.jl")
include("lwplsr_agg.jl")
include("cplsr_agg.jl")  # Use structure PlsrDa

# Local discrimination
include("lwplsrda.jl")
include("lwplslda.jl")
include("lwplsqda.jl")
include("knnda.jl")

# Variable importance (direct methods) 
include("vimp_r.jl")

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
    sourcedir,
    ensure_df, ensure_mat, list, vcol, vrow, rmcols, rmrows,
    pnames, psize,
    mweights,
    colmeans, colvars, colvars!, 
    center, center!, scale, scale!,
    matcov, matB, matW, 
    summ, aggstat, 
    tab, tabnum,
    dummy,
    recodcat2num, recodnum2cla,
    replacebylev,
    aov1,
   # Pre-processing
    snv, snv!, detrend, detrend!, fdif, fdif!,
    mavg, mavg!, mavg_runmean, mavg_runmean!,
    savgk, savgol, savgol!,
    interpl,
    rmgap, rmgap!,
    eposvd,
    # Pca
    pcasvd, pcasvd!, pcaeigen, pcaeigen!, pcaeigenk, pcaeigenk!,
    kpca,
    scordis, odis,
    # Regression
    mlr, mlr!, mlrchol, mlrchol!, mlrpinv, mlrpinv!, mlrpinv_n, mlrpinv_n!,
    mlrvec!, mlrvec,
    plskern, plskern!, plsnipals, plsnipals!,
    cglsr, cglsr!, 
    rr, rr!, rrchol, rrchol!,   
    krr, kplsr, kplsr!, dkplsr, dkplsr!,
    plsr_agg, plsr_agg!,
    svmr,   
    dfplsr_cg, aicplsr,
    #
    baggr, baggr_vi, baggr_oob,
    vimp_perm_r, vimp_chisq_r, vimp_aov_r, 
    #
    treer_xgb, rfr_xgb, xgboostr, vimp_xgb,
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
    lda, qda, 
    plsrda, kplsrda,
    rrda, krrda,
    plslda, plsqda,
    # Local Discrimination
    lwplsrda,
    lwplslda, lwplsqda,
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
    # Not exported since surchage
    # summary => Base.summary

end # End-Module




