module Jchemo

using LinearAlgebra

include("my_f.jl")
include("auxiliary.jl")
include("plskern.jl")

export my_f
export mweights, colmeans, center, center!
export list, ensure_mat, row, col
export plskern, plskern!
export summary, transform, coef, predict 

end # module
