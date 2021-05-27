module Jchemo

using LinearAlgebra, DataFrames

include("auxiliary.jl")
include("lmr.jl")
include("plskern.jl")
include("my_f.jl")

export mweights, colmeans, center, center!
export list, ensure_mat, row, col
export lmrqr!, lmrqr, lmrchol!, lmrchol, lmrpinv!, lmrpinv, lmrpinv2!, lmrpinv2, lmrvec!, lmrvec
export plskern!, plskern
export transform, coef, predict, predict_beta
export my_f
#export summary


end # Module
