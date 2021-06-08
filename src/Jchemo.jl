module Jchemo  # Start-Module

using LinearAlgebra, Statistics
using DataFrames
using Distances
using NearestNeighbors

include("auxiliary.jl")
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

export ensure_mat, list, row, col, rmrows, rmcols
export mweights
export colmeans, colvars, colvars!
export center, center!, scale, scale!
export snv, snv!, fdif, fdif!, fdif2
export lmrqr, lmrqr!, lmrchol, lmrchol!, lmrpinv, lmrpinv!, lmrpinv2, lmrpinv2!
export lmrvec!, lmrvec
export plskern, plskern!
export plsr_agg, plsr_agg! 
export locw, locwlv
export transform, coef, predict, predict_beta   
## summary: not exported since this is a surchage of Base.summary
export residreg, residcla, msep, rmsep, bias, sep, err, mse
export mpars
export gridscore, gridscorelv
export getknn, wdist
export euclsq, mahsq, mahsqchol


end # End-Module




