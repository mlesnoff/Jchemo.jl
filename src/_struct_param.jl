############---- Preprocessing

Base.@kwdef mutable struct ParDetrendpol
    degree::Int = 1  
end 

Base.@kwdef mutable struct ParDetrendlo{Q <: AbstractFloat}      
    span::Q = 0.75
    degree::Int = 2               
end 

Base.@kwdef mutable struct ParDetrendasls{Q <: AbstractFloat}
    lb::Q = 10
    p::Q = 1e-3 
    tol::Q = 1e-6    # Baeck et al 2015 p.253 
    maxit::Int = 50 
    verbose::Bool = false      
end 

Base.@kwdef mutable struct ParDetrendairpls{Q <: AbstractFloat}
    lb::Q = 10
    maxit::Int = 20 
    verbose::Bool = false      
end 

Base.@kwdef mutable struct ParDetrendarpls{Q <: AbstractFloat}
    lb::Q = 10
    tol::Q = 1e-6    # Baeck et al 2015 p.253  
    maxit::Int = 50 
    verbose::Bool = false      
end 

Base.@kwdef mutable struct ParFdif
    npoint::Int = 3  
end 

Base.@kwdef mutable struct ParInterpl{Q <: AbstractFloat}
    wl::Union{Vector{Q}, UnitRange{Q}, StepRangeLen{Q}} = range(1, 10; length = 3)
    wlfin::Union{Vector{Q}, UnitRange{Q}, StepRangeLen{Q}} = range(1, 10; length = 3)
end 

Base.@kwdef mutable struct ParMavg
    npoint::Int = 5  
end 

Base.@kwdef mutable struct ParEmsc
    degree::Int = 1 
end 

Base.@kwdef mutable struct ParSavgol
    npoint::Int = 11  
    deriv::Int = 1     
    degree::Int = 2  
end 

Base.@kwdef mutable struct ParRmgap
    indexcol::Union{Int, Vector{Int}} = 10 
    npoint::Int = 5  
end 

Base.@kwdef mutable struct ParSnv
    centr::Bool = true
    scal::Bool = true  
end 

############---- Dimension reduction

Base.@kwdef mutable struct ParPca
    nlv::Int = 1   
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPcanipals{Q <: AbstractFloat}
    nlv::Int = 1     
    gs::Bool = true   
    tol::Q = 1e-8   
    maxit::Int = 200     
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPcapp
    nlv::Int = 1
    nsim::Int = 2000  
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPcaout{Q <: AbstractFloat}
    nlv::Int = 1
    prm::Q = .3  
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParSpca{Q <: AbstractFloat}
    nlv::Int = 1 
    meth::Symbol = :soft 
    algo::Symbol = :shen  # masked in the API
    defl::Symbol = :v
    nvar::Union{Int, Vector{Int}} = 1  
    tol::Q = 1e-8 
    maxit::Int = 200   
    scal::Symbol = :none                   
end 

Base.@kwdef mutable struct ParKern{Q <: AbstractFloat}  
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1                        
end 

Base.@kwdef mutable struct ParKpca{Q <: AbstractFloat}
    nlv::Int = 1 
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1        
    scal::Symbol = :none                   
end 

Base.@kwdef mutable struct ParCovsel
    nlv::Int = 1    
    scal::Symbol = :none
end 

Base.@kwdef mutable struct ParRp{Q <: AbstractFloat}
    nlv::Int = 1    
    meth::Symbol = :gauss  
    s::Q = 1. 
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParUmap{Q <: AbstractFloat}
    psamp::Q = 1.     
    nlv::Int = 1
    metric = Distances.Euclidean()
    n_neighbors::Int = 15 
    min_dist::Q = .1                 
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParFda{Q <: AbstractFloat}
    nlv::Int = 1     
    lb::Q = 1e-6 
    prior::Union{Symbol, Vector{Q}} = :prop                  
    scal::Symbol = :none 
end 

## Multiblock

Base.@kwdef mutable struct ParBlock
    centr::Bool = false   
    scal::Symbol = :none  
    bscal::Symbol = :none   
end 

Base.@kwdef mutable struct ParCpca{Q <: AbstractFloat}
    nlv::Int = 1     
    bscal::Symbol = :none   
    tol::Q = 1e-8   
    maxit::Int = 200    
    scal::Symbol = :none  
end 

Base.@kwdef mutable struct ParCca{Q <: AbstractFloat}
    nlv::Int = 1     
    bscal::Symbol = :none   
    tau::Q = 1e-8
    tol::Q = 1e-8   
    maxit::Int = 200    
    scal::Symbol = :none  
end 

Base.@kwdef mutable struct ParCcawold{Q <: AbstractFloat}
    nlv::Int = 1     
    bscal::Symbol = :none   
    tau::Q = 1e-8
    tol::Q = 1e-8   
    maxit::Int = 200    
    scal::Symbol = :none  
end 

Base.@kwdef mutable struct ParPls2bl    # plscan, plstuck
    nlv::Int = 1     
    bscal::Symbol = :none   
    scal::Symbol = :none  
end 

Base.@kwdef mutable struct ParRasvd{Q <: AbstractFloat}
    nlv::Int = 1     
    bscal::Symbol = :none   
    tau::Q = 1e-8
    scal::Symbol = :none  
end 

############---- Regression

Base.@kwdef mutable struct ParMlr
    noint::Bool = false                      
end 

Base.@kwdef mutable struct ParNipals{Q <: AbstractFloat}
    tol::Q = 1e-8 
    maxit::Int = 200                    
end 

Base.@kwdef mutable struct ParSnipals{Q <: AbstractFloat}
    meth::Symbol = :soft
    nvar::Union{Int, Vector{Int}} = 1    
    tol::Q = 1e-8 
    maxit::Int = 200                    
end 

Base.@kwdef mutable struct ParPlsr    # except plswold
    nlv::Int = 1                    
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPlswold{Q <: AbstractFloat}    
    nlv::Int = 1     
    tol::Q = 1e-8
    maxit::Int = 200  
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPlsravgunif
    nlv::AbstractVector{Int} = 1:1                    
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPlsravg
    algo::Symbol = :unif                   
    nlv::AbstractVector{Int} = 1:1                    
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPlsrout{Q <: AbstractFloat}
    nlv::Int = 1 
    prm::Q = .3         
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParKplsr{Q <: AbstractFloat} 
    nlv::Int = 1 
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1 
    tol::Q = 1e-8   
    maxit::Int = 200            
    scal::Symbol = :none                   
end 

Base.@kwdef mutable struct ParCglsr
    nlv::Int = 1 
    gs::Bool = true       
    filt::Bool = true              
    scal::Symbol = :none 
end  

Base.@kwdef mutable struct ParRrr{Q <: AbstractFloat}
    nlv::Int = 1 
    tau::Q = 1e-8       
    tol::Q = 1e-8  
    maxit::Int = 200     
    scal::Symbol = :none                   
end 

Base.@kwdef mutable struct ParPcr
    nlv::Int = 1    
    scal::Symbol = :none                   
end

Base.@kwdef mutable struct ParRr{Q <: AbstractFloat}   
    lb::Q = 1e-6                    
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParKrr{Q <: AbstractFloat} 
    lb::Q = 1e-6
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1                       
    scal::Symbol = :none 
end 

##

Base.@kwdef mutable struct ParSplsr{Q <: AbstractFloat}
    nlv::Int = 1 
    meth::Symbol = :soft
    nvar::Union{Int, Vector{Int}} = 1
    tol::Q = 1e-8 # used when Y (n, q) (snipals)
    maxit::Int = 200              # used when Y (n, q) (snipals)
    scal::Symbol = :none                   
end 

Base.@kwdef mutable struct ParSpcr  # same ParSpca
end 

##

Base.@kwdef mutable struct ParLoessr{Q <: AbstractFloat}      
    span::Q = 0.75
    degree::Int = 2               
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParKnn{Q <: AbstractFloat}    # knnr, knnda                
    metric::Symbol = :eucl                  
    h::Q = Inf                        
    k::Int = 1                              
    criw::Q = 4.                       
    squared::Bool = false                   
    tolw::Q = 1e-4                               
    scal::Symbol = :none 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwmlr{Q <: AbstractFloat}    # lwmlr, lwmlrda                
    metric::Symbol = :eucl                  
    h::Q = Inf                        
    k::Int = 1                              
    criw::Q = 4.                       
    squared::Bool = false                   
    tolw::Q = 1e-4                               
    scal::Symbol = :none 
    store::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwplsr{Q <: AbstractFloat}  
    nlvdis::Int = 0                         
    metric::Symbol = :eucl                  
    k::Int = 1                              
    h::Q = Inf                        
    criw::Q = 4.                       
    squared::Bool = false                   
    tolw::Q = 1e-4                    
    nlv::Int =  1     
    scal::Symbol = :none
    store::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwplsravg{Q <: AbstractFloat}
    nlvdis::Int = 0                         
    metric::Symbol = :eucl                  
    k::Int = 1                              
    h::Q = Inf                        
    criw::Q = 4.                       
    squared::Bool = false                   
    tolw::Q = 1e-4                    
    nlv::AbstractVector{Int} = 1:1     
    scal::Symbol = :none
    store::Bool = false 
    verbose::Bool = false                   
end 

## Svm, Trees

Base.@kwdef mutable struct ParSvm{Q <: AbstractFloat}    # svmr, svmda
    kern::Symbol = :krbf    
    gamma::Q = 1.  
    coef0::Q = 0.   
    degree::Int = 1    
    cost::Q = 1.  
    epsilon::Q = .1 
    scal::Symbol = :none         
end 

Base.@kwdef mutable struct ParTree{Q <: AbstractFloat}    # treer, treeda
    n_subfeatures::Q = 0.  
    max_depth::Int = -1   
    min_samples_leaf::Int = 5       
    min_samples_split::Int = 5
    scal::Symbol = :none              
end 

Base.@kwdef mutable struct ParRf{Q <: AbstractFloat}    # rfr, rfda
    n_trees::Int = 10   
    partial_sampling::Q = .7  
    n_subfeatures::Q = 0   
    max_depth::Int = -1    
    min_samples_leaf::Int = 5  
    min_samples_split::Int = 5    
    mth::Bool = true  
    scal::Symbol = :none         
end 

## Multiblock

Base.@kwdef mutable struct ParMbplsr{Q <: AbstractFloat}
    nlv::Int = 1 
    bscal::Symbol = :none   
    tol::Q = 1e-8  # mbplswest
    maxit::Int = 200               # mbplswest     
    scal::Symbol = :none  
end 

Base.@kwdef mutable struct ParSoplsr
    nlv::Union{Int, Vector{Int}} = 1     
    scal::Symbol = :none  
end 

Base.@kwdef mutable struct ParRosaplsr
    nlv::Int = 1     
    scal::Symbol = :none  
end 

############---- Discrimination

Base.@kwdef mutable struct ParRda{Q <: AbstractFloat}
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0.
    lb::Q = 1e-6
    simpl::Bool = false 
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParDmnorm
    simpl::Bool = false 
end 

Base.@kwdef mutable struct ParDmkern{Q <: AbstractFloat}
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
end 

Base.@kwdef mutable struct ParLda{Q <: AbstractFloat}
    prior::Union{Symbol, Vector{Q}} = :prop                     
end 

Base.@kwdef mutable struct ParQda{Q <: AbstractFloat}
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0.                      
end 


Base.@kwdef mutable struct ParKdeda{Q <: AbstractFloat}
    prior::Union{Symbol, Vector{Q}} = :prop
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
end 

##

Base.@kwdef mutable struct ParMlrda{Q <: AbstractFloat}
    prior::Union{Symbol, Vector{Q}} = :prop                     
end 

Base.@kwdef mutable struct ParRrda{Q <: AbstractFloat}
    lb::Q = 1e-6
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParPlsda{Q <: AbstractFloat}    # plsrda, plslda
    nlv::Int = 1
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParPlsqda{Q <: AbstractFloat}
    nlv::Int = 1
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0. 
    scal::Symbol = :none                 
end 

Base.@kwdef mutable struct ParPlskdeda{Q <: AbstractFloat}
    nlv::Int = 1
    prior::Union{Symbol, Vector{Q}} = :prop
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParSplsda{Q <: AbstractFloat}    # splsrda, splslda
    nlv::Int = 1
    meth::Symbol = :soft 
    nvar::Union{Int, Vector{Int}} = 1  
    prior::Union{Symbol, Vector{Q}} = :prop   
    tol::Q = 1e-8 
    maxit::Int = 200   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParSplsqda{Q <: AbstractFloat}
    nlv::Int = 1
    meth::Symbol = :soft 
    nvar::Union{Int, Vector{Int}} = 1  
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0.   
    tol::Q = 1e-8 
    maxit::Int = 200    
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParSplskdeda{Q <: AbstractFloat}
    nlv::Int = 1
    meth::Symbol = :soft 
    nvar::Union{Int, Vector{Int}} = 1  
    prior::Union{Symbol, Vector{Q}} = :prop
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
    tol::Q = 1e-8 
    maxit::Int = 200   
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParKrrda{Q <: AbstractFloat}
    lb::Q = 1e-6
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1      
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParKplsda{Q <: AbstractFloat}    # kplsrda, kplslda
    nlv::Int = 1
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1      
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParKplsqda{Q <: AbstractFloat}
    nlv::Int = 1
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1 
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0.    
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParKplskdeda{Q <: AbstractFloat}
    nlv::Int = 1
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1 
    prior::Union{Symbol, Vector{Q}} = :prop
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
    scal::Symbol = :none 
end 

## 

Base.@kwdef mutable struct ParLwplsda{Q <: AbstractFloat}    # lwplsrda, lwplslda 
    nlvdis::Int = 0                         
    metric::Symbol = :eucl                  
    h::Q = Inf                        
    k::Int = 1                              
    criw::Q = 4.                       
    squared::Bool = false                   
    tolw::Q = 1e-4                    
    prior::Union{Symbol, Vector{Q}} = :prop
    nlv::Int = 1      
    scal::Symbol = :none 
    store::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwplsqda{Q <: AbstractFloat}    
    nlvdis::Int = 0                         
    metric::Symbol = :eucl                  
    h::Q = Inf                        
    k::Int = 1                              
    criw::Q = 4.                       
    squared::Bool = false                   
    tolw::Q = 1e-4                    
    prior::Union{Symbol, Vector{Q}} = :prop
    nlv::Int = 1 
    alpha::Q = 0.        
    scal::Symbol = :none 
    store::Bool = false 
    verbose::Bool = false                   
end 

## Multiblock

Base.@kwdef mutable struct ParMbplsda{Q <: AbstractFloat}  # mbplsrda, mbplslda
    nlv::Int = 1
    bscal::Symbol = :none   
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParMbplsqda{Q <: AbstractFloat}  
    nlv::Int = 1
    bscal::Symbol = :none   
    prior::Union{Symbol, Vector{Q}} = :prop 
    alpha::Q = 0.   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParMbplskdeda{Q <: AbstractFloat}  
    nlv::Int = 1
    bscal::Symbol = :none   
    prior::Union{Symbol, Vector{Q}} = :prop 
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1.  
    scal::Symbol = :none                    
end 

## Occ 

Base.@kwdef mutable struct ParOcc{Q <: AbstractFloat}    # occsd, occod
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
end 

Base.@kwdef mutable struct ParOccsdod{Q <: AbstractFloat}
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
    gamma::Q = .5
    fscal::Function = madv
end 

Base.@kwdef mutable struct ParOccdds{Q <: AbstractFloat}
    fcentr::Function = meanv
    fscal::Function = stdv
    alpha::Q = .05 
end 

Base.@kwdef mutable struct ParOccstah{Q <: AbstractFloat} 
    nlv::Int = 500
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
    scal::Symbol = :none 
    seed::Union{Nothing, Int} = nothing                   
end 

Base.@kwdef mutable struct ParOccknn{Q <: AbstractFloat}
    nsamp::Int = 100
    metric::Symbol = :eucl                                       
    k::Int = 1     
    algo::Function = sum
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
    scal::Symbol = :none 
    seed::Union{Nothing, Int} = nothing                  
end 


