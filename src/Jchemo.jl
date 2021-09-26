module Jchemo  # Start-Module

using LinearAlgebra, Statistics, Random
using CairoMakie
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


include("utility.jl") 
include("preprocessing.jl") 
include("rmgap.jl")
include("aov1.jl")
include("plots.jl")
include("matW.jl")

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

# Variable importance 
include("imp_r.jl")

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
    mweights,
    colmeans, colvars, colvars!, 
    center, center!, scale, scale!,
    matcov, matB, matW, 
    summ, aggstat, 
    tab, tabnum,
    dummy,
    recodcat2num, recodnum2cla,
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
    imp_perm_r, imp_chisq_r, imp_aov_r, 
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
    krbf, kpol,
    # Graphics
    plotsp
    # Not exported since surchage
    # summary => Base.summary

end # End-Module




