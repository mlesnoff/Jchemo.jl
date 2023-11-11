Base.@kwdef struct Par
    nlv::Int = 1
    lb::Float64 = .01
    ## 
    kern::Symbol = :krbf
    gamma::Float64 = 1. 
    degree::Int = 3
    coef0::Float64 = 0. 
    cost::Float64 = 1.
    ##
    gs::Bool = true
    epsilon::Float64 = .1
    tol::Float64 = sqrt(eps(1.))
    maxit::Int = 200
    scal::Bool = false
end

struct Weight{T <: AbstractFloat}
    w::Vector{T} 
end



