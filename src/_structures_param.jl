############---- Preprocessing

Base.@kwdef mutable struct ParAsls
    lb::Float64 = 10
    p::Float64 = 1e-3 
    tol::Float64 = sqrt(eps(1.))    
    maxit::Int = 50 
    verbose::Bool = false      
end 

Base.@kwdef mutable struct ParAirpls
    lb::Float64 = 10
    maxit::Int = 20 
    verbose::Bool = false      
end 

Base.@kwdef mutable struct ParDetrend
    degree::Int = 1  
end 

Base.@kwdef mutable struct ParFdif
    npoint::Int = 3  
end 

Base.@kwdef mutable struct ParInterpl
    wl::Union{Vector, UnitRange, StepRangeLen} = range(1, 10; length = 3)
    wlfin::Union{Vector, UnitRange, StepRangeLen} = range(1, 10; length = 3)
end 

Base.@kwdef mutable struct ParMavg
    npoint::Int = 5  
end 

Base.@kwdef mutable struct ParRmgap
    indexcol::Union{Int, Vector{Int}} = 10 
    npoint::Int = 5  
end 

Base.@kwdef mutable struct ParSavgol
    npoint::Int = 11  
    degree::Int = 2  
    deriv::Int = 1     
end 

Base.@kwdef mutable struct ParSnv
    centr::Bool = true
    scal::Bool = true  
end 

############---- Dimension reduction

Base.@kwdef mutable struct ParPca
    nlv::Int = 1   
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParPcanipals
    nlv::Int = 1     
    gs::Bool = true   
    tol::Float64 = sqrt(eps(1.))    
    maxit::Int = 200     
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParPcapp
    nlv::Int = 1
    nsim::Int = 2000  
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParPcaout
    nlv::Int = 1
    prm::Float64 = .3  
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParSpca
    nlv::Int = 1 
    meth::Symbol = :soft 
    delta::Float64 = 0.   
    nvar::Union{Int, Vector{Int}} = 1  
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParKern  
    gamma::Float64 = 1.  
    coef0::Float64 = 0.
    degree::Int = 1                        
end 

Base.@kwdef mutable struct ParKpca
    nlv::Int = 1 
    kern::Symbol = :krbf     
    gamma::Float64 = 1.  
    coef0::Float64 = 0.
    degree::Int = 1        
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParRp
    nlv::Int = 1    
    meth::Symbol = :gauss  
    s::Float64 = 1. 
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParUmap
    nlv::Int = 1     
    n_neighbors::Int = 15 
    min_dist::Float64 = .1                 
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParFda
    nlv::Int = 1     
    lb::Float64 = 1e-6 
    prior::Union{Symbol, Vector{Float64}} = :unif                  
    scal::Bool = false 
end 

## Multiblock

Base.@kwdef mutable struct ParBlock
    bscal::Symbol = :none   
    centr::Bool = false   
    scal::Bool = false  
end 

Base.@kwdef mutable struct ParMbpca
    nlv::Int = 1     
    bscal::Symbol = :none   
    tol::Float64 = sqrt(eps(1.))    
    maxit::Int = 200    
    scal::Bool = false  
end 

Base.@kwdef mutable struct ParCca
    nlv::Int = 1     
    bscal::Symbol = :none   
    tau::Float64 = 1e-8
    tol::Float64 = sqrt(eps(1.))    
    maxit::Int = 200    
    scal::Bool = false  
end 

Base.@kwdef mutable struct ParCcawold
    nlv::Int = 1     
    bscal::Symbol = :none   
    tau::Float64 = 1e-8
    tol::Float64 = sqrt(eps(1.))    
    maxit::Int = 200    
    scal::Bool = false  
end 

Base.@kwdef mutable struct ParPls2bl    # plscan, plstuck
    nlv::Int = 1     
    bscal::Symbol = :none   
    scal::Bool = false  
end 

Base.@kwdef mutable struct ParRasvd
    nlv::Int = 1     
    bscal::Symbol = :none   
    tau::Float64 = 1e-8
    scal::Bool = false  
end 

############---- Regression

Base.@kwdef mutable struct ParNipals
    tol::Float64 = sqrt(eps(1.))  
    maxit::Int = 200                    
end 

Base.@kwdef mutable struct ParSnipals
    delta::Float64 = 0.   
    nvar::Union{Int, Vector{Int}} = 1    
    tol::Float64 = sqrt(eps(1.))  
    maxit::Int = 200                    
end 

Base.@kwdef mutable struct ParMlr
    noint::Bool = false                      
end 

Base.@kwdef mutable struct ParPlsr    # {plskern, ..., plsravg} except plswold
    nlv::Union{Int, Vector{Int}, UnitRange} = 1                    
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParPlswold    
    nlv::Union{Int, Vector{Int}, UnitRange} = 1     
    tol::Float64 = sqrt(eps(1.)) 
    maxit::Int = 200  
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParCglsr
    nlv::Int = 1 
    gs::Bool = true       
    filt::Bool = true              
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParPcr
    nlv::Int = 1    
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParRrr
    nlv::Int = 1 
    tau::Float64 = 1e-8       
    tol::Float64 = sqrt(eps(1.))   
    maxit::Int = 200     
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParPlsrout
    nlv::Int = 1 
    prm::Float64 = .3         
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParSplsr
    nlv::Int = 1 
    meth::Symbol = :soft 
    delta::Float64 = 0.   
    nvar::Union{Int, Vector{Int}} = 1  
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParKplsr
    nlv::Int = 1 
    kern::Symbol = :krbf     
    gamma::Float64 = 1.  
    coef0::Float64 = 0.
    degree::Int = 1 
    tol::Float64 = sqrt(eps(1.))   
    maxit::Int = 200            
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

##

Base.@kwdef mutable struct ParKnn    # knnr, lwmlr, knnda, lwmlrda                
    metric::Symbol = :eucl                  
    h::Float64 = Inf                        
    k::Int = 1                              
    criw::Float64 = 4                       
    squared::Bool = false                   
    tolw::Float64 = 1e-4                               
    scal::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwplsr    # lwplsr, lwplsravg
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

## Svm, Trees

Base.@kwdef mutable struct ParSvm    # svmr, svmda
    kern::Symbol = :krbf    
    gamma::Float64 = 1.  
    coef0::Float64 = 0.   
    degree::Int = 1    
    cost::Float64 = 1.  
    epsilon::Float64 = .1 
    scal::Bool = false         
end 

Base.@kwdef mutable struct ParTree    # treer, treeda
    n_subfeatures::Float64 = 0  
    max_depth::Int = -1   
    min_samples_leaf::Int = 5       
    min_samples_split::Int = 5
    scal::Bool = false              
end 

Base.@kwdef mutable struct ParRf    # rfr, rfda
    n_trees::Int = 10   
    partial_sampling::Float64 = .7  
    n_subfeatures::Float64 = 0   
    max_depth::Int = -1    
    min_samples_leaf::Int = 5  
    min_samples_split::Int = 5    
    mth::Bool = true  
    scal::Bool = false         
end 

## Multiblock

Base.@kwdef mutable struct ParMbplsr
    nlv::Int = 1 
    bscal::Symbol = :none   
    tol::Float64 = sqrt(eps(1.))   # mbplswest
    maxit::Int = 200               # mbplswest     
    scal::Bool = false  
end 

Base.@kwdef mutable struct ParSoplsr
    nlv::Union{Int, Vector{Int}} = 1     
    scal::Bool = false  
end 

############---- Discrimination

Base.@kwdef mutable struct ParDmnorm
    mu::Union{Nothing, Vector} = nothing 
    S::Union{Nothing, Matrix} = nothing     
    simpl::Bool = false 
end 

Base.@kwdef mutable struct ParDmkern
    h::Union{Nothing, Float64, Vector{Float64}} = nothing  
    a::Float64 = 1. 
end 

Base.@kwdef mutable struct ParLda
    prior::Union{Symbol, Vector{Float64}} = :unif                     
end 

Base.@kwdef mutable struct ParQda
    prior::Union{Symbol, Vector{Float64}} = :unif
    alpha::Float64 = 0.                      
end 

Base.@kwdef mutable struct ParRda
    prior::Union{Symbol, Vector{Float64}} = :unif
    alpha::Float64 = 0.
    lb::Float64 = 1e-6
    simpl::Bool = false 
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParKdeda
    prior::Union{Symbol, Vector{Float64}} = :unif
    h::Union{Nothing, Float64, Vector{Float64}} = nothing  
    a::Float64 = 1. 
end 

##

Base.@kwdef mutable struct ParMlrda
    prior::Union{Symbol, Vector{Float64}} = :unif                     
end 

##

Base.@kwdef mutable struct ParPlsda    # plsrda, plslda
    nlv::Int = 1
    prior::Union{Symbol, Vector{Float64}} = :unif   
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParPlsqda
    nlv::Int = 1
    prior::Union{Symbol, Vector{Float64}} = :unif
    alpha::Float64 = 0. 
    scal::Bool = false                 
end 

Base.@kwdef mutable struct ParPlskdeda
    nlv::Int = 1
    prior::Union{Symbol, Vector{Float64}} = :unif
    h::Union{Nothing, Float64, Vector{Float64}} = nothing  
    a::Float64 = 1. 
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParRrda
    lb::Float64 = 1e-6
    prior::Union{Symbol, Vector{Float64}} = :unif   
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParSplsda    # splsrda, splslda
    nlv::Int = 1
    meth::Symbol = :soft 
    delta::Float64 = 0.   
    nvar::Union{Int, Vector{Int}} = 1  
    prior::Union{Symbol, Vector{Float64}} = :unif   
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParSplsqda
    nlv::Int = 1
    meth::Symbol = :soft 
    delta::Float64 = 0.   
    nvar::Union{Int, Vector{Int}} = 1  
    prior::Union{Symbol, Vector{Float64}} = :unif
    alpha::Float64 = 0.    
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParSplskdeda
    nlv::Int = 1
    meth::Symbol = :soft 
    delta::Float64 = 0.   
    nvar::Union{Int, Vector{Int}} = 1  
    prior::Union{Symbol, Vector{Float64}} = :unif
    h::Union{Nothing, Float64, Vector{Float64}} = nothing  
    a::Float64 = 1. 
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParKplsda    # kplsrda, kplslda
    nlv::Int = 1
    kern::Symbol = :krbf     
    gamma::Float64 = 1.  
    coef0::Float64 = 0.
    degree::Int = 1      
    prior::Union{Symbol, Vector{Float64}} = :unif   
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParKplsqda
    nlv::Int = 1
    kern::Symbol = :krbf     
    gamma::Float64 = 1.  
    coef0::Float64 = 0.
    degree::Int = 1 
    prior::Union{Symbol, Vector{Float64}} = :unif
    alpha::Float64 = 0.    
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParKplskdeda
    nlv::Int = 1
    kern::Symbol = :krbf     
    gamma::Float64 = 1.  
    coef0::Float64 = 0.
    degree::Int = 1 
    prior::Union{Symbol, Vector{Float64}} = :unif
    h::Union{Nothing, Float64, Vector{Float64}} = nothing  
    a::Float64 = 1. 
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParKrrda
    lb::Float64 = 1e-6
    kern::Symbol = :krbf     
    gamma::Float64 = 1.  
    coef0::Float64 = 0.
    degree::Int = 1      
    prior::Union{Symbol, Vector{Float64}} = :unif   
    scal::Bool = false                    
end 

## 

Base.@kwdef mutable struct ParLwplsda    # lwplsrda, lwplslda 
    nlvdis::Int = 0                         
    metric::Symbol = :eucl                  
    h::Float64 = Inf                        
    k::Int = 1                              
    criw::Float64 = 4                       
    squared::Bool = false                   
    tolw::Float64 = 1e-4                    
    nlv::Int = 1 
    prior::Union{Symbol, Vector{Float64}} = :unif       
    scal::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwplsqda    
    nlvdis::Int = 0                         
    metric::Symbol = :eucl                  
    h::Float64 = Inf                        
    k::Int = 1                              
    criw::Float64 = 4                       
    squared::Bool = false                   
    tolw::Float64 = 1e-4                    
    nlv::Int = 1 
    prior::Union{Symbol, Vector{Float64}} = :unif
    alpha::Float64 = 0.        
    scal::Bool = false 
    verbose::Bool = false                   
end 

## Multiblock

Base.@kwdef mutable struct ParMbplsda  # mbplsrda, mbplslda
    nlv::Int = 1
    bscal::Symbol = :none   
    prior::Union{Symbol, Vector{Float64}} = :unif   
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParMbplsqda  
    nlv::Int = 1
    bscal::Symbol = :none   
    prior::Union{Symbol, Vector{Float64}} = :unif 
    alpha::Float64 = 0.   
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParMbplskdeda  
    nlv::Int = 1
    bscal::Symbol = :none   
    prior::Union{Symbol, Vector{Float64}} = :unif 
    h::Union{Nothing, Float64, Vector{Float64}} = nothing  
    a::Float64 = 1.  
    scal::Bool = false                    
end 

## Occ 

Base.@kwdef mutable struct ParOut
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParOcc    # occsd, occod, occsdod
    mcut::Symbol = :mad   
    risk::Float64 = .025  
    cri::Float64 = 3. 
end 

Base.@kwdef mutable struct ParOccstah 
    nlv::Int = 500
    mcut::Symbol = :mad   
    risk::Float64 = .025  
    cri::Float64 = 3. 
    scal::Bool = false                    
end 

