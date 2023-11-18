Base.@kwdef mutable struct Par
    nlv::Union{Int, Vector{Int}, UnitRange} = 1
    lb::Float64 = 1e-5
    ##
    bscal::Symbol = :none
    ##
    noint::Bool = false
    ##
    tau::Float64 = 1e-8
    ## 
    kern::Symbol = :krbf
    gamma::Float64 = 1. 
    degree::Int = 3
    coef0::Float64 = 0. 
    cost::Float64 = 1.
    epsilon::Float64 = .1
    ##
    n_trees::Int = 10
    partial_sampling::Float64 = .7
    n_subfeatures::Float64 = 0   # internally rounded/set to Int 
    max_depth::Int = -1
    min_samples_leaf::Int = 5
    min_samples_split::Int = 5
    mth::Bool = true
    ##
    prior::Symbol = :unif
    ##
    gs::Bool = true
    filt::Bool = true
    tol::Float64 = sqrt(eps(1.))
    maxit::Int = 200
    scal::Bool = false
    ##
    alpha_aic::Float64 = 2.
    meth_sp::Symbol = :soft
    delta::Float64 = 0.
    nvar::Union{Int, Vector{Int}} = 1
end

struct Weight{T <: AbstractFloat}
    w::Vector{T} 
end
