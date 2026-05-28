############---- Preprocessing

Base.@kwdef mutable struct ParDetrendpol
    degree::Int = 1  
end 

Base.@kwdef mutable struct ParDetrendlo{Q <: Float64}      
    span::Q = 0.75
    degree::Int = 2               
end 

Base.@kwdef mutable struct ParDetrendasls{Q <: Float64}
    lb::Q = 10
    p::Q = 1e-3 
    tol::Q = 1e-6    # Baeck et al 2015 p.253 
    maxit::Int = 50 
    verbose::Bool = false      
end 

Base.@kwdef mutable struct ParDetrendairpls{Q <: Float64}
    lb::Q = 10
    maxit::Int = 20 
    verbose::Bool = false      
end 

Base.@kwdef mutable struct ParDetrendarpls{Q <: Float64}
    lb::Q = 10
    tol::Q = 1e-6    # Baeck et al 2015 p.253  
    maxit::Int = 50 
    verbose::Bool = false      
end 

Base.@kwdef mutable struct ParEmsc
    degree::Int = 1 
end 

Base.@kwdef mutable struct ParFdif{Q <: Float64}
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
    deriv::Int = 1     
    degree::Int = 2  
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

Base.@kwdef mutable struct ParPcanipals{Q <: Float64}
    nlv::Int = 1     
    gs::Bool = true   
    tol::Q = sqrt(eps(1.))    
    maxit::Int = 200     
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParPcapp
    nlv::Int = 1
    nsim::Int = 2000  
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParPcaout{Q <: Float64}
    nlv::Int = 1
    prm::Q = .3  
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParSpca{Q <: Float64}
    nlv::Int = 1 
    meth::Symbol = :soft 
    algo::Symbol = :shen  # masked in the API
    defl::Symbol = :v
    nvar::Union{Int, Vector{Int}} = 1  
    tol::Q = sqrt(eps(1.))  
    maxit::Int = 200   
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParKern{Q <: Float64}  
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1                        
end 

Base.@kwdef mutable struct ParKpca{Q <: Float64}
    nlv::Int = 1 
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1        
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParCovsel
    nlv::Int = 1    
    scal::Bool = false
end 

Base.@kwdef mutable struct ParRp{Q <: Float64}
    nlv::Int = 1    
    meth::Symbol = :gauss  
    s::Q = 1. 
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParUmap{Q <: Float64}
    psamp::Q = 1.     
    nlv::Int = 1
    metric = Distances.Euclidean()
    n_neighbors::Int = 15 
    min_dist::Q = .1                 
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParFda{Q <: Float64}
    nlv::Int = 1     
    lb::Q = 1e-6 
    prior::Union{Symbol, Vector{Q}} = :prop                  
    scal::Bool = false 
end 

## Multiblock

Base.@kwdef mutable struct ParBlock
    centr::Bool = false   
    scal::Bool = false  
    bscal::Symbol = :none   
end 

Base.@kwdef mutable struct ParMbpca{Q <: Float64}
    nlv::Int = 1     
    bscal::Symbol = :none   
    tol::Q = sqrt(eps(1.))    
    maxit::Int = 200    
    scal::Bool = false  
end 

Base.@kwdef mutable struct ParCca{Q <: Float64}
    nlv::Int = 1     
    bscal::Symbol = :none   
    tau::Q = 1e-8
    tol::Q = sqrt(eps(1.))    
    maxit::Int = 200    
    scal::Bool = false  
end 

Base.@kwdef mutable struct ParCcawold{Q <: Float64}
    nlv::Int = 1     
    bscal::Symbol = :none   
    tau::Q = 1e-8
    tol::Q = sqrt(eps(1.))    
    maxit::Int = 200    
    scal::Bool = false  
end 

Base.@kwdef mutable struct ParPls2bl    # plscan, plstuck
    nlv::Int = 1     
    bscal::Symbol = :none   
    scal::Bool = false  
end 

Base.@kwdef mutable struct ParRasvd{Q <: Float64}
    nlv::Int = 1     
    bscal::Symbol = :none   
    tau::Q = 1e-8
    scal::Bool = false  
end 

############---- Regression

Base.@kwdef mutable struct ParNipals{Q <: Float64}
    tol::Q = sqrt(eps(1.))  
    maxit::Int = 200                    
end 

Base.@kwdef mutable struct ParSnipals{Q <: Float64}
    meth::Symbol = :soft
    nvar::Union{Int, Vector{Int}} = 1    
    tol::Q = sqrt(eps(1.))  
    maxit::Int = 200                    
end 

Base.@kwdef mutable struct ParMlr
    noint::Bool = false                      
end 

Base.@kwdef mutable struct ParRr{Q <: Float64}   
    lb::Q = 1e-6                    
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParPcr
    nlv::Int = 1    
    scal::Bool = false                   
end

Base.@kwdef mutable struct ParPlsr    # {plskern, ..., plsravg} except plswold
    nlv::Union{Int, Vector{Int}, UnitRange} = 1                    
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParPlswold{Q <: Float64}    
    nlv::Union{Int, Vector{Int}, UnitRange} = 1     
    tol::Q = sqrt(eps(1.)) 
    maxit::Int = 200  
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParCglsr
    nlv::Int = 1 
    gs::Bool = true       
    filt::Bool = true              
    scal::Bool = false 
end  

Base.@kwdef mutable struct ParRrr{Q <: Float64}
    nlv::Int = 1 
    tau::Q = 1e-8       
    tol::Q = sqrt(eps(1.))   
    maxit::Int = 200     
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParPlsrout{Q <: Float64}
    nlv::Int = 1 
    prm::Q = .3         
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParSpcr  # same ParSpca
end 

Base.@kwdef mutable struct ParSplsr{Q <: Float64}
    nlv::Int = 1 
    meth::Symbol = :soft
    nvar::Union{Int, Vector{Int}} = 1
    tol::Q = sqrt(eps(1.))   # used when Y (n, q) (snipals)
    maxit::Int = 200         # used when Y (n, q) (snipals)
    scal::Bool = false                   
end 

Base.@kwdef mutable struct ParKrr{Q <: Float64} 
    lb::Q = 1e-6
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1                       
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParKplsr{Q <: Float64}
    nlv::Int = 1 
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1 
    tol::Q = sqrt(eps(1.))   
    maxit::Int = 200            
    scal::Bool = false                   
end 

##

Base.@kwdef mutable struct ParKnn{Q <: Float64}    # knnr, knnda                
    metric::Symbol = :eucl                  
    h::Q = Inf                        
    k::Int = 1                              
    criw::Q = 4                       
    squared::Bool = false                   
    tolw::Q = 1e-4                               
    scal::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwmlr{Q <: Float64}    # lwmlr, lwmlrda                
    metric::Symbol = :eucl                  
    h::Q = Inf                        
    k::Int = 1                              
    criw::Q = 4                       
    squared::Bool = false                   
    tolw::Q = 1e-4                               
    scal::Bool = false 
    store::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwplsr{Q <: Float64}    # lwplsr, lwplsravg
    nlvdis::Int = 0                         
    metric::Symbol = :eucl                  
    k::Int = 1                              
    h::Q = Inf                        
    criw::Q = 4.                       
    squared::Bool = false                   
    tolw::Q = 1e-4                    
    nlv::Union{Int, Vector{Int}, UnitRange} = 1     
    scal::Bool = false
    store::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLoessr{Q <: Float64}      
    span::Q = 0.75
    degree::Int = 2               
    scal::Bool = false 
end 

## Svm, Trees

Base.@kwdef mutable struct ParSvm{Q <: Float64}    # svmr, svmda
    kern::Symbol = :krbf    
    gamma::Q = 1.  
    coef0::Q = 0.   
    degree::Int = 1    
    cost::Q = 1.  
    epsilon::Q = .1 
    scal::Bool = false         
end 

Base.@kwdef mutable struct ParTree{Q <: Float64}    # treer, treeda
    n_subfeatures::Q = 0  
    max_depth::Int = -1   
    min_samples_leaf::Int = 5       
    min_samples_split::Int = 5
    scal::Bool = false              
end 

Base.@kwdef mutable struct ParRf{Q <: Float64}    # rfr, rfda
    n_trees::Int = 10   
    partial_sampling::Q = .7  
    n_subfeatures::Q = 0   
    max_depth::Int = -1    
    min_samples_leaf::Int = 5  
    min_samples_split::Int = 5    
    mth::Bool = true  
    scal::Bool = false         
end 

## Multiblock

Base.@kwdef mutable struct ParMbplsr{Q <: Float64}
    nlv::Int = 1 
    bscal::Symbol = :none   
    tol::Q = sqrt(eps(1.))   # mbplswest
    maxit::Int = 200               # mbplswest     
    scal::Bool = false  
end 

Base.@kwdef mutable struct ParSoplsr
    nlv::Union{Int, Vector{Int}} = 1     
    scal::Bool = false  
end 

############---- Discrimination

Base.@kwdef mutable struct ParRda{Q <: Float64}
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0.
    lb::Q = 1e-6
    simpl::Bool = false 
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParDmnorm
    simpl::Bool = false 
end 

Base.@kwdef mutable struct ParDmkern{Q <: Float64}
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
end 

Base.@kwdef mutable struct ParLda #{Q <: Float64}
    prior::Union{Symbol, Vector{Float64}} = :prop                     
end 

Base.@kwdef mutable struct ParQda{Q <: Float64}
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0.                      
end 

Base.@kwdef mutable struct ParKdeda{Q <: Float64}
    prior::Union{Symbol, Vector{Q}} = :prop
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
end 

##

Base.@kwdef mutable struct ParMlrda{Q <: Float64}
    prior::Union{Symbol, Vector{Q}} = :prop                     
end 

Base.@kwdef mutable struct ParRrda{Q <: Float64}
    lb::Q = 1e-6
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParPlsda #{Q <: Float64}    # plsrda, plslda
    nlv::Int = 1
    prior::Union{Symbol, Vector{Float64}} = :prop   
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParPlsqda{Q <: Float64}
    nlv::Int = 1
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0. 
    scal::Bool = false                 
end 

Base.@kwdef mutable struct ParPlskdeda{Q <: Float64}
    nlv::Int = 1
    prior::Union{Symbol, Vector{Q}} = :prop
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParSplsda{Q <: Float64}    # splsrda, splslda
    nlv::Int = 1
    meth::Symbol = :soft 
    nvar::Union{Int, Vector{Int}} = 1  
    prior::Union{Symbol, Vector{Q}} = :prop   
    tol::Q = sqrt(eps(1.))  
    maxit::Int = 200   
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParSplsqda{Q <: Float64}
    nlv::Int = 1
    meth::Symbol = :soft 
    nvar::Union{Int, Vector{Int}} = 1  
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0.   
    tol::Q = sqrt(eps(1.))  
    maxit::Int = 200    
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParSplskdeda{Q <: Float64}
    nlv::Int = 1
    meth::Symbol = :soft 
    nvar::Union{Int, Vector{Int}} = 1  
    prior::Union{Symbol, Vector{Q}} = :prop
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
    tol::Q = sqrt(eps(1.))  
    maxit::Int = 200   
    scal::Bool = false 
end 

Base.@kwdef mutable struct ParKrrda{Q <: Float64}
    lb::Q = 1e-6
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1      
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParKplsda{Q <: Float64}    # kplsrda, kplslda
    nlv::Int = 1
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1      
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParKplsqda{Q <: Float64}
    nlv::Int = 1
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1 
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0.    
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParKplskdeda{Q <: Float64}
    nlv::Int = 1
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Int = 1 
    prior::Union{Symbol, Vector{Q}} = :prop
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
    scal::Bool = false 
end 

## 

Base.@kwdef mutable struct ParLwplsda{Q <: Float64}    # lwplsrda, lwplslda 
    nlvdis::Int = 0                         
    metric::Symbol = :eucl                  
    h::Q = Inf                        
    k::Int = 1                              
    criw::Q = 4                       
    squared::Bool = false                   
    tolw::Q = 1e-4                    
    prior::Union{Symbol, Vector{Q}} = :prop
    nlv::Int = 1      
    scal::Bool = false 
    store::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwplsqda{Q <: Float64}    
    nlvdis::Int = 0                         
    metric::Symbol = :eucl                  
    h::Q = Inf                        
    k::Int = 1                              
    criw::Q = 4                       
    squared::Bool = false                   
    tolw::Q = 1e-4                    
    prior::Union{Symbol, Vector{Q}} = :prop
    nlv::Int = 1 
    alpha::Q = 0.        
    scal::Bool = false 
    store::Bool = false 
    verbose::Bool = false                   
end 

## Multiblock

Base.@kwdef mutable struct ParMbplsda{Q <: Float64}  # mbplsrda, mbplslda
    nlv::Int = 1
    bscal::Symbol = :none   
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParMbplsqda{Q <: Float64}  
    nlv::Int = 1
    bscal::Symbol = :none   
    prior::Union{Symbol, Vector{Q}} = :prop 
    alpha::Q = 0.   
    scal::Bool = false                    
end 

Base.@kwdef mutable struct ParMbplskdeda{Q <: Float64}  
    nlv::Int = 1
    bscal::Symbol = :none   
    prior::Union{Symbol, Vector{Q}} = :prop 
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1.  
    scal::Bool = false                    
end 

## Occ 

Base.@kwdef mutable struct ParOcc{Q <: Float64}    # occsd, occod
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
end 

Base.@kwdef mutable struct ParOccsdod{Q <: Float64}
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
    gamma::Q = .5
    fscal::Function = madv
end 

Base.@kwdef mutable struct ParOccdds{Q <: Float64}
    fcentr::Function = meanv
    fscal::Function = stdv
    alpha::Q = .05 
end 

Base.@kwdef mutable struct ParOccstah{Q <: Float64} 
    nlv::Int = 500
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
    scal::Bool = false 
    seed::Union{Nothing, Int} = nothing                   
end 

Base.@kwdef mutable struct ParOccknn{Q <: Float64}
    nsamp::Int = 100
    metric::Symbol = :eucl                                       
    k::Int = 1     
    algo::Function = sum
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
    scal::Bool = false 
    seed::Union{Nothing, Int} = nothing                  
end 
