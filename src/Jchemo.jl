module Jchemo  # Start-Module

using LinearAlgebra, Statistics
using Distributions
using DataFrames
using ImageFiltering
using Distances
using NearestNeighbors

include("utility.jl") ; include("center_scale.jl")
include("preprocessing.jl")
include("pcasvd.jl") ; include("pcaeigen.jl")
include("kpca.jl")
include("lmr.jl")
include("rr.jl")
include("plskern.jl") ; include("cglsr.jl") 
include("plsr_agg.jl")
include("krr.jl") ; include("kplsr.jl") ; include("dkplsr.jl")
include("locw.jl")
include("knnr.jl") ; include("lwplsr.jl")
include("lwplsr_agg.jl")
include("mpars.jl")
include("scores.jl") ; include("gridscore.jl")
include("distances.jl")
include("getknn.jl")
include("wdist.jl")
include("kernels.jl")

include("xfit.jl") ; include("scordis.jl")


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
    # Pca
    pcasvd, pcasvd!, pcaeigen, pcaeigen!, pcaeigenk, pcaeigenk!,
    kpca,
    xfit, xfit!, xresid, xresid!,
    scordis, odis,
    # Regression
    lmrqr, lmrqr!, lmrchol, lmrchol!, lmrpinv, lmrpinv!, lmrpinv_n, lmrpinv_n!,
    lmrvec!, lmrvec,
    rr, rr!, rrchol, rrchol!,   
    plskern, plskern!, cglsr, cglsr!,
    plsr_agg, plsr_agg!,
    krr, kplsr, kplsr!, dkplsr, dkplsr!,
    # Local
    locw, locwlv,
    knnr, lwplsr, lwplsr_agg,
    transform, coef, predict,
    # Validation
    residreg, residcla, msep, rmsep, bias, sep, err, mse,
    mpars,
    gridscore, gridscorelv, gridscorelb,
    # Distances
    getknn, wdist, wdist!,
    euclsq, mahsq, mahsqchol,
    krbf, kpol
    # Not exported since surchage: summary (Base.summary)

end # End-Module




