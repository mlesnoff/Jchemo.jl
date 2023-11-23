Base.@kwdef mutable struct Par
    nlv::Union{Int, Vector{Int}, UnitRange} = 1     # nb LVs
    lb::Float64 = 1e-5                              # ridge parameter "lambda"
    ##
    bscal::Symbol = :none                   # type of block scaling 
    ##
    noint::Bool = false                     # intercept in MLR models
    ##
    tau::Float64 = 1e-8                     # regularization parameter in multi-block methods
    ## 
    kern::Symbol = :krbf                    # type of kernel
    gamma::Float64 = 1.                     # kernel parameter
    degree::Int = 3                         # polynomial kernel
    coef0::Float64 = 0.                     # polynomial kernel
    cost::Float64 = 1.                      # svm
    epsilon::Float64 = .1                   # svm
    ##
    n_trees::Int = 10                       # random forest
    partial_sampling::Float64 = .7          # random forest
    n_subfeatures::Float64 = 0              # internally rounded/set to Int 
    max_depth::Int = -1                     # tree, random forest
    min_samples_leaf::Int = 5               # tree, random forest
    min_samples_split::Int = 5              # tree, random forest
    mth::Bool = true                        # multi-threading in random forest
    ##
    prior::Symbol = :unif                   # prior in DA
    ##
    gs::Bool = true                         # Gram-Schmidt orthogonalization 
    filt::Bool = true                       # cglsr
    tol::Float64 = sqrt(eps(1.))            # tolerance in Nipals
    maxit::Int = 200                        # maximal nb. iterations in Nipals 
    scal::Bool = false                      # scaling data columns
    ##
    alpha_aic::Float64 = 2.                 # aicplsr
    meth_sp::Symbol = :soft                 # sparse methods, threshold 
    delta::Float64 = 0.                     # sparse methods, threshold
    nvar::Union{Int, Vector{Int}} = 1       # sparse methods, threshold
    ##
    rpmeth::Symbol = :gauss        # rp projection method
    s::Float64 = 1.                         # rpmatli sparsity parameter  
end

struct Weight{T <: AbstractFloat}
    w::Vector{T} 
end
