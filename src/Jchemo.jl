module Jchemo  # Start-Module

using LinearAlgebra, Statistics
using DataFrames
using ImageFiltering
using Distances
using NearestNeighbors

include("utility.jl")
include("center_scale.jl")
include("preprocessing.jl")
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
    snv, snv!, fdif, fdif!,
    mavg, mavg!, mavg_runmean, mavg_runmean!,
    savgk, savgol, savgol!,
    lmrqr, lmrqr!, lmrchol, lmrchol!, lmrpinv, lmrpinv!, lmrpinv2, lmrpinv2!,
    lmrvec!, lmrvec,
    plskern, plskern!,
    plsr_agg, plsr_agg!,
    locw, locwlv,
    transform, coef, predict, predict_beta,
    ## summary: not exported since this is a surchage of Base.summary
    residreg, residcla, msep, rmsep, bias, sep, err, mse,
    mpars,
    gridscore, gridscorelv,
    getknn, wdist,
    euclsq, mahsq, mahsqchol


end # End-Module




