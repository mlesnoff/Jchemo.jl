Base.@kwdef mutable struct Par
    nlv::Int = 1
    lb::Float64 = 1e-5
    ##
    noint::Bool = false
    ## 
    kern::Symbol = :krbf
    gamma::Float64 = 1. 
    degree::Int = 3
    coef0::Float64 = 0. 
    cost::Float64 = 1.
    ##
    prior::Symbol = :unif
    ##
    gs::Bool = true
    filt::Bool = true
    epsilon::Float64 = .1
    tol::Float64 = sqrt(eps(1.))
    maxit::Int = 200
    scal::Bool = false
    ##
    meth_sp::Symbol = :soft
    delta::Float64 = 0.
    nvar::Union{Int, Vector{Int}} = 1
end

struct Weight{T <: AbstractFloat}
    w::Vector{T} 
end




