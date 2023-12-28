Base.@kwdef mutable struct Par
    nlv::Union{Int, Vector{Int}, UnitRange} = 1     # nb LVs
    lb::Float64 = 1e-6                              # ridge parameter "lambda"
    ##
    centr::Bool = false                     # centering
    scal::Bool = false                      # scaling
    ##
    npoint::Int = 5                         # Nb. points defining a window
    deriv::Int = 1                          # derivation order
    degree::Int = 1                         # degree of polynom
    wl::Union{Vector, UnitRange, StepRangeLen} = range(1, 10; length = 3)
    wlfin::Union{Vector, UnitRange, StepRangeLen} = range(1, 10; length = 3)
    ##
    noint::Bool = false                     # intercept in MLR models
    ##
    tau::Float64 = 1e-8                     # regularization parameter in multi-block methods
    bscal::Symbol = :none                   # type of block scaling 
    ## 
    kern::Symbol = :krbf                    # type of kernel
    gamma::Float64 = 1.                     # kernel parameter
    coef0::Float64 = 0.                     # coefficient in polynomial kernel
    cost::Float64 = 1.                      # svm
    epsilon::Float64 = .1                   # svm
    ##
    n_trees::Int = 10                       # nb treees in random forest
    partial_sampling::Float64 = .7          # row sampling in random forest
    n_subfeatures::Float64 = 0              # internally rounded/set to Int 
    max_depth::Int = -1                     # tree, random forest
    min_samples_leaf::Int = 5               # tree, random forest
    min_samples_split::Int = 5              # tree, random forest
    mth::Bool = true                        # multi-threading in random forest
    ##
    mreduc::Symbol = :pls                   # type of preliminary reduction
    nlvreduc::Int = 20                      # nb LVs for preliminary reduction 
    msamp::Symbol = :rand                   # method of row sampling
    psamp::Float64 = 1.                     # row sampling
    nlvdis::Int = 0                         # nb LVs for global space
    metric::Symbol = :eucl                  # metric for global space
    h::Float64 = Inf                        # shape parameter in fweight
    k::Int = 1                              # nb neighbors
    cri_w::Float64 = 4                      # coefficient for cutoff in wdist
    squared::Bool = false                   # type of curve in wdist 
    tolw::Float64 = 1e-4                    # tolerance for local weights
    verbose::Bool = false                   # print obs. indexes when prediction
    ##
    prior::Symbol = :unif                   # prior in DA
    alpha::Float64 = 0.                     # regularization in qda and rda 
    simpl::Bool = false                     # dmnorm-parameter in rda
    h_kde::Union{Nothing, Float64} = nothing  # dmkern-parameter 'h' in kdeda
    a_kde::Float64 = 1.                     # dmkern-parameter 'a' in kdeda
    ##
    freduc::Function = pcasvd               # method of dimension reduction in occsd-od-sdod
    mcut::Symbol = :mad                     # type of cutoff in occ methods
    risk::Float64 = .025                    # risk I ("alpha") for cutoff in occ methods
    cri::Float64 = 3.                       # coefficient for cutoff in occ methods
    ##
    gs::Bool = true                         # Gram-Schmidt orthogonalization 
    filt::Bool = true                       # cglsr
    tol::Float64 = sqrt(eps(1.))            # tolerance in Nipals
    maxit::Int = 200                        # maximal nb. iterations in Nipals 
    ##
    msparse::Symbol = :soft                 # threshold in sparse methods 
    delta::Float64 = 0.                     # threshold in sparse methods
    nvar::Union{Int, Vector{Int}} = 1       # threshold in sparse methods
    ##
    mrp::Symbol = :gauss                    # rp projection method
    s_li::Float64 = 1.                      # rpmatli sparsity parameter  
end 

