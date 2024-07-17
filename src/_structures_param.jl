Base.@kwdef mutable struct ParUmap
    nlv::Union{Int, Vector{Int}, UnitRange} = 1     
    n_neighbors::Int = 15 
    min_dist::Float64 = .1                 
    scal::Bool = false 
end 

###### Regression

Base.@kwdef mutable struct ParMlr
    noint::Bool = false                      
end 

Base.@kwdef mutable struct ParPlsr
    nlv::Union{Int, Vector{Int}, UnitRange} = 1     
    tol::Float64 = sqrt(eps(1.))            # plswold
    maxit::Int = 200                        # plswold                 
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParCglsr
    nlv::Union{Int, Vector{Int}, UnitRange} = 1     
    gs::Bool = true       
    filt::Bool = true              
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParPcr
    nlv::Union{Int, Vector{Int}, UnitRange} = 1     
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParRrr
    nlv::Union{Int, Vector{Int}, UnitRange} = 1
    tau::Float64 = 1e-8       
    tol::Float64 = sqrt(eps(1.))   
    maxit::Int = 200     
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParOut
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParPlsrOut
    nlv::Union{Int, Vector{Int}, UnitRange} = 1      
    prm::Float64 = .3                       # plsrout                    
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParSplsr
    nlv::Union{Int, Vector{Int}, UnitRange} = 1
    msparse::Symbol = :soft 
    delta::Float64 = 0.   
    nvar::Union{Int, Vector{Int}} = 1  
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParKern  
    gamma::Float64 = 1.  
    coef0::Float64 = 0.
    degree::Int = 1                        
end 

Base.@kwdef mutable struct ParKplsr
    nlv::Union{Int, Vector{Int}, UnitRange} = 1
    kern::Symbol = :krbf     
    gamma::Float64 = 1.  
    coef0::Float64 = 0.
    degree::Int = 1        
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParRr   
    lb::Float64 = 1e-6                    
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParKrr 
    lb::Float64 = 1e-6
    kern::Symbol = :krbf     
    gamma::Float64 = 1.  
    coef0::Float64 = 0.
    degree::Int = 1                       
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParLwmlr                 
    metric::Symbol = :eucl                  
    h::Float64 = Inf                        
    k::Int = 1                              
    criw::Float64 = 4                       
    squared::Bool = false                   
    tolw::Float64 = 1e-4                               
    scal::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwplsr
    nlvdis::Int = 0                         
    metric::Symbol = :eucl                  
    h::Float64 = Inf                        
    k::Int = 1                              
    criw::Float64 = 4                       
    squared::Bool = false                   
    tolw::Float64 = 1e-4                    
    nlv::Union{Int, Vector{Int}, UnitRange} = 1               
    scal::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParSvmr
    kern::Symbol = :krbf    
    gamma::Float64 = 1.  
    coef0::Float64 = 0.   
    degree::Int = 1    
    cost::Float64 = 1.  
    epsilon::Float64 = .1 
    scal::Bool = false         
end 

Base.@kwdef mutable struct ParTreer
    n_subfeatures::Float64 = 0  
    max_depth::Int = -1   
    min_samples_leaf::Int = 5       
    min_samples_split::Int = 5
    scal::Bool = false              
end 

Base.@kwdef mutable struct ParRfr
    n_trees::Int = 10   
    partial_sampling::Float64 = .7  
    n_subfeatures::Float64 = 0   
    max_depth::Int = -1    
    min_samples_leaf::Int = 5  
    min_samples_split::Int = 5    
    mth::Bool = true  
    scal::Bool = false         
end 

Base.@kwdef mutable struct ParBlock
    bscal::Symbol = :none   
    centr::Bool = false   
    scal::Bool = false  
end 

Base.@kwdef mutable struct ParMbplsr
    nlv::Union{Int, Vector{Int}, UnitRange} = 1     
    bscal::Symbol = :none   
    tol::Float64 = sqrt(eps(1.))   # mbplswest
    maxit::Int = 200               # mbplswest     
    scal::Bool = false  
end 

Base.@kwdef mutable struct ParSoplsr
    nlv::Union{Int, Vector{Int}, UnitRange} = 1     
    scal::Bool = false  
end 

###### Preprocessing

Base.@kwdef mutable struct ParCalds
    nlv::Union{Int, Vector{Int}, UnitRange} = 1     
    scal::Bool = false  
end 














Base.@kwdef mutable struct Par
    nlv::Union{Int, Vector{Int}, UnitRange} = 1     # nb LVs
    lb::Float64 = 1e-6                              # ridge parameter "lambda"
    nsim::Int = 2000                                # nb additional simulated directions for PP
    prm::Float64 = .3                               # proportion of removed data in 'pcaout'
    n_neighbors::Int = 15 
    min_dist::Float64 = .1
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
    indexcol::Union{Int, Vector{Int}} = 10  # index of columns in 'rmgap'
    fun::Function = plskern                 # model used in DS and PDS
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
    msamp::Symbol = :rand                   # method of row sampling
    psamp::Float64 = 1.                     # proportion of row sampling
    ##
    nlvdis::Int = 0                         # nb LVs for global space
    metric::Symbol = :eucl                  # metric for global space
    h::Float64 = Inf                        # shape parameter in fweight
    k::Int = 1                              # nb neighbors
    criw::Float64 = 4                       # coefficient for cutoff in wdist
    squared::Bool = false                   # type of curve in wdist 
    tolw::Float64 = 1e-4                    # tolerance for local weights
    verbose::Bool = false                   # print obs. indexes when prediction
    ##
    prior::Union{Symbol, Vector{Float64}} = :unif              # prior in DA
    alpha::Float64 = 0.                     # regularization in 'qda' and 'rda' 
    simpl::Bool = false                     # dmnorm-parameter in 'rda'
    h_kde::Union{Nothing, Float64, Vector{Float64}} = nothing  # dmkern-parameter 'h' in kdeda
    a_kde::Float64 = 1.                     # dmkern-parameter 'a' in kdeda
    ##
    freduc::Function = pcasvd               # method of dimension reduction in occsd-od-sdod
    mcut::Symbol = :mad                     # type of cutoff in occ methods
    risk::Float64 = .025                    # risk I ("alpha") for cutoff in occ methods
    cri::Float64 = 3.                       # coefficient for cutoff in occ methods
    ##
    gs::Bool = true                         # Gram-Schmidt orthogonalization 
    filt::Bool = true                       # 'cglsr'
    tol::Float64 = sqrt(eps(1.))            # tolerance in Nipals
    maxit::Int = 200                        # maximal nb. iterations in Nipals 
    ##
    msparse::Symbol = :soft                 # threshold in sparse methods 
    delta::Float64 = 0.                     # threshold in sparse methods
    nvar::Union{Int, Vector{Int}} = 1       # threshold in sparse methods
    ##
    mrp::Symbol = :gauss                    # 'rp' projection method
    s_li::Float64 = 1.                      # 'rpmatli' sparsity parameter  
end 


