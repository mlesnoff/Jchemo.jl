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
include("rr.jl")
include("plskern.jl")
include("plsr_agg.jl")
include("krr.jl")
include("locw.jl")
include("lwplsr.jl")
include("mpars.jl")
include("scores.jl")
include("gridscore.jl")
include("distances.jl")
include("getknn.jl")
include("wdist.jl")
include("kernels.jl")

export 
    ensure_mat, list, vcol, vrow, rmcols, rmrows,
    mweights,
    colmeans, colvars, colvars!,
    center, center!, scale, scale!,
    snv, snv!, detrend, detrend!, fdif, fdif!,
    mavg, mavg!, mavg_runmean, mavg_runmean!,
    savgk, savgol, savgol!,
    pcasvd, pcasvd!, pcaeigen, pcaeigen!, pcaeigenk, pcaeigenk!,
    lmrqr, lmrqr!, lmrchol, lmrchol!, lmrpinv, lmrpinv!, lmrpinv_n, lmrpinv_n!,
    lmrvec!, lmrvec,
    rrsvd, rrsvd!, rrchol, rrchol!,   
    plskern, plskern!,
    plsr_agg, plsr_agg!,
    krrsvd,
    locw, locwlv,
    lwplsr,
    transform, coef, predict,
    residreg, residcla, msep, rmsep, bias, sep, err, mse,
    mpars,
    gridscore, gridscorelv,
    getknn, wdist, wdist!,
    euclsq, mahsq, mahsqchol,
    krbf, kpol
    ## Not exported since surchage: summary (Base.summary)

end # End-Module




