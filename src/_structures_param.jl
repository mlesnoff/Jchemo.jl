Base.@kwdef struct Par
    nlv::Int = 1
    lb::Float64 = 1e-5
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
end

struct Weight{T <: AbstractFloat}
    w::Vector{T} 
end



