module Jchemo

using LinearAlgebra, DataFrames

include("auxiliary.jl")
include("lmr.jl")
include("plskern.jl")
include("mpars.jl")
include("scores.jl")
include("gridscore.jl")
include("my_f.jl")

export mweights, colmeans, center, center!
export list, ensure_mat, row, col
export lmrqr!, lmrqr, lmrchol!, lmrchol, lmrpinv!, lmrpinv, lmrpinv2!, lmrpinv2, lmrvec!, lmrvec
export plskern!, plskern
export transform, coef, predict, predict_beta
export mpars
export residreg, residcla, msep, rmsep, bias, sep, err, mse
export gridscore, gridscorelv
#export my_f

end # Module
