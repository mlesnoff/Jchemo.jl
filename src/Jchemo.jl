module Jchemo  # Start-Module

using LinearAlgebra, Statistics, Random
using Distributions
using StatsBase    # sample
using DataFrames
using ImageFiltering
using Distances
using NearestNeighbors
using DecisionTree

include("utility.jl") ; include("center_scale.jl")
include("preprocessing.jl")
include("rmgap.jl")

include("pcasvd.jl") ; include("pcaeigen.jl")
include("kpca.jl")

include("lmr.jl")
include("rr.jl")
include("plskern.jl") ; include("plsnipals.jl") 
include("cglsr.jl") ; include("aicplsr.jl") ; include("wshenk.jl")  
include("krr.jl") ; include("kplsr.jl") ; include("dkplsr.jl")
include("plsr_agg.jl")

include("treer.jl")
include("baggr.jl") ; include("baggr_vi.jl")
include("gboostr.jl") ; include("boostr.jl")

include("xfit.jl") ; include("scordis.jl")

include("locw.jl")
include("knnr.jl") ; include("lwplsr.jl")
include("lwplsr_agg.jl")

# Validation
include("mpars.jl")
include("scores.jl")
include("gridscore.jl")
include("segm.jl") ; include("gridcv.jl")

include("distances.jl")
include("getknn.jl")
include("wdist.jl")
include("kernels.jl")

export 
    # Utilities
    ensure_mat, list, vcol, vrow, rmcols, rmrows,
    mweights,
    colmeans, colvars, colvars!,
    center, center!, scale, scale!,
    # Pre-processing
    snv, snv!, detrend, detrend!, fdif, fdif!,
    mavg, mavg!, mavg_runmean, mavg_runmean!,
    savgk, savgol, savgol!,
    rmgap, rmgap!,
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
    treer,
    baggr, baggr_oob, baggr_vi,
    gboostr, boostr, boostrw,
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
    # Distances
    getknn, wdist, wdist!,
    euclsq, mahsq, mahsqchol,
    krbf, kpol
    # Not exported since surchage
    # summary => Base.summary

end # End-Module




