module Jchemo

using LinearAlgebra, Statistics
using DataFrames
using NearestNeighbors

include("auxiliary.jl")
include("lmr.jl")
include("plskern.jl")
include("plsragg.jl")
include("mpars.jl")
include("scores.jl")
include("gridscore.jl")
include("getknn.jl") ; include("wdist.jl")
include("locw.jl")

export mweights, colmeans, center, center!
export list, ensure_mat, row, col, rmrow
export lmrqr!, lmrqr, lmrchol!, lmrchol, lmrpinv!, lmrpinv, lmrpinv2!, lmrpinv2, lmrvec!, lmrvec
export plskern!, plskern
export plsragg!, plsragg
export transform, coef, predict, predict_beta
export mpars
export residreg, residcla, msep, rmsep, bias, sep, err, mse
export gridscore, gridscorelv
export getknn, wdist, locw, locwlv
#export my_f

end # Module
