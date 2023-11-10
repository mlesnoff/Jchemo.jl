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
    epsilon::Float64 = .1
    tol::Float64 = 1.5e-8
    maxit::Int = 100
    scal::Bool = false
end

