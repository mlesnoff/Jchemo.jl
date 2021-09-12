module Jchemo  # Start-Module

using LinearAlgebra, Statistics, Random
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
using Distributed

include("utility.jl") 
include("preprocessing.jl") 
include("rmgap.jl")
include("aov1.jl")

include("pcasvd.jl") ; include("pcaeigen.jl")
include("kpca.jl")

include("lmr.jl")
include("rr.jl")
include("plskern.jl") ; include("plsnipals.jl") 
include("cglsr.jl") ; include("aicplsr.jl") ; include("wshenk.jl")  
include("krr.jl") ; include("kplsr.jl") ; include("dkplsr.jl")
include("plsr_agg.jl")

include("svmr.jl")

include("baggr.jl") ; include("baggr_util.jl")
include("gboostr.jl") ; include("boostr.jl")

include("treer_xgb.jl")

include("xfit.jl") ; include("scordis.jl")

include("locw.jl")
include("knnr.jl") ; include("lwplsr.jl")
include("lwplsr_agg.jl")

# Var imp 
include("var_imp.jl")

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
    ensure_mat, list, vcol, vrow, rmcols, rmrows,
    mweights,
    colmeans, colvars, colvars!,
    center, center!, scale, scale!,
    tab, tabnum,
    dummy,
    recod2cla,
    aov1,
    varimp_chisq, varimp_aov, 
    # Pre-processing
    snv, snv!, detrend, detrend!, fdif, fdif!,
    mavg, mavg!, mavg_runmean, mavg_runmean!,
    savgk, savgol, savgol!,
    rmgap, rmgap!,
    interpl,
    eposvd,
    # Pca
    pcasvd, pcasvd!, pcaeigen, pcaeigen!, pcaeigenk, pcaeigenk!,
    kpca,
    scordis, odis,
    # Regression
    lmr, lmrqr!, lmrchol, lmrchol!, lmrpinv, lmrpinv!, lmrpinv_n, lmrpinv_n!,
    lmrvec!, lmrvec,
    rr, rr!, rrchol, rrchol!,   
    plskern, plskern!, plsnipals, plsnipals!,
    cglsr, cglsr!, dfplsr_cg, aicplsr,
    krr, kplsr, kplsr!, dkplsr, dkplsr!,
    plsr_agg, plsr_agg!,
    #
    svmr,
    # 
    baggr, baggr_vi, baggr_oob,
    #
    treer_xgb, rfr_xgb, xgboostr,
    #
    xfit, xfit!, xresid, xresid!,
    # Local
    locw, locwlv,
    knnr, lwplsr, lwplsr_agg,
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
    krbf, kpol
    # Not exported since surchage
    # summary => Base.summary

end # End-Module




