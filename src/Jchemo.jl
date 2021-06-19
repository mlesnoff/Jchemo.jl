module Jchemo  # Start-Module

using LinearAlgebra, Statistics
using DataFrames
using ImageFiltering
using Distances
using NearestNeighbors

include("utility.jl")
include("center_scale.jl")
include("preprocessing.jl")
include("pcasvd.jl")
include("pcaeigen.jl")
include("lmr.jl")
include("plskern.jl")
include("plsr_agg.jl")
include("locw.jl")
include("mpars.jl")
include("scores.jl")
include("gridscore.jl")
include("distances.jl")
include("getknn.jl")
include("wdist.jl")

export 
    ensure_mat, list, vcol, vrow, rmcols, rmrows,
    mweights,
    colmeans, colvars, colvars!,
    center, center!, scale, scale!,
    snv, snv!, detrend, detrend!, fdif, fdif!,
    mavg, mavg!, mavg_runmean, mavg_runmean!,
    savgk, savgol, savgol!,
    pcasvd, pcasvd!, pcaeigen, pcaeigen!, pcaeigenk, pcaeigenk!,
    lmrqr, lmrqr!, lmrchol, lmrchol!, lmrpinv, lmrpinv!, lmrpinvn, lmrpinvn!,
    lmrvec!, lmrvec,
    plskern, plskern!,
    plsr_agg, plsr_agg!,
    locw, locwlv,
    transform, coef, predict,
    ## summary: not exported since this is a surchage of Base.summary
    residreg, residcla, msep, rmsep, bias, sep, err, mse,
    mpars,
    gridscore, gridscorelv,
    getknn, wdist, wdist!,
    euclsq, mahsq, mahsqchol
end # End-Module




