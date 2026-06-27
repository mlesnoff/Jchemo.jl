############---- Preprocessing

Base.@kwdef mutable struct ParDetrendpol
    degree::Signed = 1  
end 

Base.@kwdef mutable struct ParDetrendlo{Q <: Float}      
    span::Q = 0.75
    degree::Signed = 2               
end 

Base.@kwdef mutable struct ParDetrendasls{Q <: Float}
    lb::Q = 10
    p::Q = 1e-3 
    tol::Q = 1e-6    # Baeck et al 2015 p.253 
    maxit::Signed = 50 
    verbose::Bool = false      
end 

Base.@kwdef mutable struct ParDetrendairpls{Q <: Float}
    lb::Q = 10
    maxit::Signed = 20 
    verbose::Bool = false      
end 

Base.@kwdef mutable struct ParDetrendarpls{Q <: Float}
    lb::Q = 10
    tol::Q = 1e-6    # Baeck et al 2015 p.253  
    maxit::Signed = 50 
    verbose::Bool = false      
end 

Base.@kwdef mutable struct ParFdif
    npoint::Signed = 3  
end 

Base.@kwdef mutable struct ParInterpl{Q <: Float}
    wl::Union{Vector{Q}, UnitRange{Q}, StepRangeLen{Q}} = range(1, 10; length = 3)
    wlfin::Union{Vector{Q}, UnitRange{Q}, StepRangeLen{Q}} = range(1, 10; length = 3)
end 

Base.@kwdef mutable struct ParMavg
    npoint::Signed = 5  
end 

Base.@kwdef mutable struct ParEmsc
    degree::Signed = 1 
end 

Base.@kwdef mutable struct ParSavgol
    npoint::Signed = 11  
    deriv::Signed = 1     
    degree::Signed = 2  
end 

Base.@kwdef mutable struct ParSnv
    centr::Bool = true
    scal::Bool = true  
end 

Base.@kwdef mutable struct ParRmgap
    indexcol::Union{Signed, Vector{<: Signed}} = 10 
    npoint::Signed = 5  
end 

Base.@kwdef mutable struct ParScale
    scal::Symbol = :none
end 

############---- Dimension reduction

Base.@kwdef mutable struct ParNipals{Q <: Float}
    tol::Q = 1e-8 
    maxit::Signed = 200                    
end 

Base.@kwdef mutable struct ParSnipals{Q <: Float}
    meth::Symbol = :soft
    nvar::Union{Signed, Vector{<: Signed}} = 1    
    tol::Q = 1e-8 
    maxit::Signed = 200                    
end 

Base.@kwdef mutable struct ParPca
    nlv::Signed = 1   
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPcanipals{Q <: Float}
    nlv::Signed = 1     
    gs::Bool = true   
    tol::Q = 1e-8   
    maxit::Signed = 200     
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPcapp
    nlv::Signed = 1
    nsim::Signed = 2000  
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPcaout{Q <: Float}
    nlv::Signed = 1
    prm::Q = .3  
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParSpca{Q <: Float}
    nlv::Signed = 1 
    meth::Symbol = :soft 
    algo::Symbol = :shen  # masked in the API
    defl::Symbol = :v
    nvar::Union{Signed, Vector{<: Signed}} = 1  
    tol::Q = 1e-8 
    maxit::Signed = 200   
    scal::Symbol = :none                   
end 

Base.@kwdef mutable struct ParKern{Q <: Float}  
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Signed = 1                        
end 

Base.@kwdef mutable struct ParKpca{Q <: Float}
    nlv::Signed = 1 
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Signed = 1        
    scal::Symbol = :none                   
end 

Base.@kwdef mutable struct ParRp{Q <: Float}
    nlv::Signed = 1    
    meth::Symbol = :gauss  
    s::Q = 1. 
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParUmap{Q <: Float}
    psamp::Q = 1.     
    nlv::Signed = 1
    metric = Distances.Euclidean()
    n_neighbors::Signed = 15 
    min_dist::Q = .1                 
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParFda{Q <: Float}
    nlv::Signed = 1     
    lb::Q = 1e-6 
    prior::Union{Symbol, Vector{Q}} = :prop                  
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParCovsel
    nlv::Signed = 1    
    scal::Symbol = :none
end 

## Multiblock

Base.@kwdef mutable struct ParBlock
    centr::Bool = false   
    scal::Symbol = :none  
    bscal::Symbol = :none   
end 

Base.@kwdef mutable struct ParCpca{Q <: Float}
    nlv::Signed = 1     
    bscal::Symbol = :none   
    tol::Q = 1e-8   
    maxit::Signed = 200    
    scal::Symbol = :none  
end 

Base.@kwdef mutable struct ParCca{Q <: Float}
    nlv::Signed = 1     
    bscal::Symbol = :none   
    tau::Q = 1e-8
    tol::Q = 1e-8   
    maxit::Signed = 200    
    scal::Symbol = :none  
end 

Base.@kwdef mutable struct ParCcawold{Q <: Float}
    nlv::Signed = 1     
    bscal::Symbol = :none   
    tau::Q = 1e-8
    tol::Q = 1e-8   
    maxit::Signed = 200    
    scal::Symbol = :none  
end 

Base.@kwdef mutable struct ParPls2bl    # plscan, plstuck
    nlv::Signed = 1     
    bscal::Symbol = :none   
    scal::Symbol = :none  
end 

Base.@kwdef mutable struct ParRasvd{Q <: Float}
    nlv::Signed = 1     
    bscal::Symbol = :none   
    tau::Q = 1e-8
    scal::Symbol = :none  
end 

############---- Regression

Base.@kwdef mutable struct ParMlr
    noint::Bool = false                      
end 

Base.@kwdef mutable struct ParPlsr    # except plswold
    nlv::Signed = 1                    
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPlswold{Q <: Float}    
    nlv::Signed = 1     
    tol::Q = 1e-8
    maxit::Signed = 200  
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPlsravgunif
    nlv::AbstractVector{<: Signed} = 1:1                    
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPlsravg
    algo::Symbol = :unif                   
    nlv::AbstractVector{<: Signed} = 1:1                    
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParPlsrout{Q <: Float}
    nlv::Signed = 1 
    prm::Q = .3         
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParKplsr{Q <: Float} 
    nlv::Signed = 1 
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Signed = 1 
    tol::Q = 1e-8   
    maxit::Signed = 200            
    scal::Symbol = :none                   
end 

Base.@kwdef mutable struct ParCglsr
    nlv::Signed = 1 
    gs::Bool = true       
    filt::Bool = true              
    scal::Symbol = :none 
end  

Base.@kwdef mutable struct ParRrr{Q <: Float}
    nlv::Signed = 1 
    tau::Q = 1e-8       
    tol::Q = 1e-8  
    maxit::Signed = 200     
    scal::Symbol = :none                   
end 

Base.@kwdef mutable struct ParPcr
    nlv::Signed = 1    
    scal::Symbol = :none                   
end

Base.@kwdef mutable struct ParRr{Q <: Float}   
    lb::Q = 1e-6                    
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParKrr{Q <: Float} 
    lb::Q = 1e-6
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Signed = 1                       
    scal::Symbol = :none 
end 

##

Base.@kwdef mutable struct ParSplsr{Q <: Float}
    nlv::Signed = 1 
    meth::Symbol = :soft
    nvar::Union{Signed, Vector{<: Signed}} = 1
    tol::Q = 1e-8 # used when Y (n, q) (snipals)
    maxit::Signed = 200              # used when Y (n, q) (snipals)
    scal::Symbol = :none                   
end 

Base.@kwdef mutable struct ParSpcr  # same ParSpca
end 

##

Base.@kwdef mutable struct ParLoessr{Q <: Float}      
    span::Q = 0.75
    degree::Signed = 2               
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParKnn{Q <: Float}    # knnr, knnda                
    metric::Symbol = :eucl                  
    h::Q = Inf                        
    k::Signed = 1                              
    criw::Q = 4.                       
    squared::Bool = false                   
    tolw::Q = 1e-4                               
    scal::Symbol = :none 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwmlr{Q <: Float}    # lwmlr, lwmlrda                
    metric::Symbol = :eucl                  
    h::Q = Inf                        
    k::Signed = 1                              
    criw::Q = 4.                       
    squared::Bool = false                   
    tolw::Q = 1e-4                               
    scal::Symbol = :none 
    store::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwplsr{Q <: Float}  
    nlvdis::Signed = 0                         
    metric::Symbol = :eucl                  
    k::Signed = 1                              
    h::Q = Inf                        
    criw::Q = 4.                       
    squared::Bool = false                   
    tolw::Q = 1e-4                    
    nlv::Signed =  1     
    scal::Symbol = :none
    store::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwplsravg{Q <: Float}
    nlvdis::Signed = 0                         
    metric::Symbol = :eucl                  
    k::Signed = 1                              
    h::Q = Inf                        
    criw::Q = 4.                       
    squared::Bool = false                   
    tolw::Q = 1e-4                    
    nlv::AbstractVector{<: Signed} = 1:1     
    scal::Symbol = :none
    store::Bool = false 
    verbose::Bool = false                   
end 

## Svm, Trees

Base.@kwdef mutable struct ParSvm{Q <: Float}    # svmr, svmda
    kern::Symbol = :krbf    
    gamma::Q = 1.  
    coef0::Q = 0.   
    degree::Signed = 1    
    cost::Q = 1.  
    epsilon::Q = .1 
    scal::Symbol = :none         
end 

Base.@kwdef mutable struct ParTree{Q <: Float}    # treer, treeda
    n_subfeatures::Q = 0.  
    max_depth::Signed = -1   
    min_samples_leaf::Signed = 5       
    min_samples_split::Signed = 5
    scal::Symbol = :none              
end 

Base.@kwdef mutable struct ParRf{Q <: Float}    # rfr, rfda
    n_trees::Signed = 10   
    partial_sampling::Q = .7  
    n_subfeatures::Q = 0   
    max_depth::Signed = -1    
    min_samples_leaf::Signed = 5  
    min_samples_split::Signed = 5    
    mth::Bool = true  
    scal::Symbol = :none         
end 

## Multiblock

Base.@kwdef mutable struct ParMbplsr{Q <: Float}
    nlv::Signed = 1 
    bscal::Symbol = :none   
    tol::Q = 1e-8  # mbplswest
    maxit::Signed = 200               # mbplswest     
    scal::Symbol = :none  
end 

Base.@kwdef mutable struct ParSoplsr
    nlv::Union{Signed, Vector{<: Signed}} = 1     
    scal::Symbol = :none  
end 

Base.@kwdef mutable struct ParRosaplsr
    nlv::Signed = 1     
    scal::Symbol = :none  
end 

############---- Discrimination

Base.@kwdef mutable struct ParRda{Q <: Float}
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0.
    lb::Q = 1e-6
    simpl::Bool = false 
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParDmnorm
    simpl::Bool = false 
end 

Base.@kwdef mutable struct ParDmkern{Q <: Float}
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
end 

Base.@kwdef mutable struct ParLda{Q <: Float}
    prior::Union{Symbol, Vector{Q}} = :prop                     
end 

Base.@kwdef mutable struct ParQda{Q <: Float}
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0.                      
end 


Base.@kwdef mutable struct ParKdeda{Q <: Float}
    prior::Union{Symbol, Vector{Q}} = :prop
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
end 

##

Base.@kwdef mutable struct ParMlrda{Q <: Float}
    prior::Union{Symbol, Vector{Q}} = :prop                     
end 

Base.@kwdef mutable struct ParRrda{Q <: Float}
    lb::Q = 1e-6
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParPlsda{Q <: Float}    # plsrda, plslda
    nlv::Signed = 1
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParPlsqda{Q <: Float}
    nlv::Signed = 1
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0. 
    scal::Symbol = :none                 
end 

Base.@kwdef mutable struct ParPlskdeda{Q <: Float}
    nlv::Signed = 1
    prior::Union{Symbol, Vector{Q}} = :prop
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParSplsda{Q <: Float}    # splsrda, splslda
    nlv::Signed = 1
    meth::Symbol = :soft 
    nvar::Union{Signed, Vector{<: Signed}} = 1  
    prior::Union{Symbol, Vector{Q}} = :prop   
    tol::Q = 1e-8 
    maxit::Signed = 200   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParSplsqda{Q <: Float}
    nlv::Signed = 1
    meth::Symbol = :soft 
    nvar::Union{Signed, Vector{<: Signed}} = 1  
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0.   
    tol::Q = 1e-8 
    maxit::Signed = 200    
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParSplskdeda{Q <: Float}
    nlv::Signed = 1
    meth::Symbol = :soft 
    nvar::Union{Signed, Vector{<: Signed}} = 1  
    prior::Union{Symbol, Vector{Q}} = :prop
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
    tol::Q = 1e-8 
    maxit::Signed = 200   
    scal::Symbol = :none 
end 

Base.@kwdef mutable struct ParKrrda{Q <: Float}
    lb::Q = 1e-6
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Signed = 1      
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParKplsda{Q <: Float}    # kplsrda, kplslda
    nlv::Signed = 1
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Signed = 1      
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParKplsqda{Q <: Float}
    nlv::Signed = 1
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Signed = 1 
    prior::Union{Symbol, Vector{Q}} = :prop
    alpha::Q = 0.    
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParKplskdeda{Q <: Float}
    nlv::Signed = 1
    kern::Symbol = :krbf     
    gamma::Q = 1.  
    coef0::Q = 0.
    degree::Signed = 1 
    prior::Union{Symbol, Vector{Q}} = :prop
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1. 
    scal::Symbol = :none 
end 

## 

Base.@kwdef mutable struct ParLwplsda{Q <: Float}    # lwplsrda, lwplslda 
    nlvdis::Signed = 0                         
    metric::Symbol = :eucl                  
    h::Q = Inf                        
    k::Signed = 1                              
    criw::Q = 4.                       
    squared::Bool = false                   
    tolw::Q = 1e-4                    
    prior::Union{Symbol, Vector{Q}} = :prop
    nlv::Signed = 1      
    scal::Symbol = :none 
    store::Bool = false 
    verbose::Bool = false                   
end 

Base.@kwdef mutable struct ParLwplsqda{Q <: Float}    
    nlvdis::Signed = 0                         
    metric::Symbol = :eucl                  
    h::Q = Inf                        
    k::Signed = 1                              
    criw::Q = 4.                       
    squared::Bool = false                   
    tolw::Q = 1e-4                    
    prior::Union{Symbol, Vector{Q}} = :prop
    nlv::Signed = 1 
    alpha::Q = 0.        
    scal::Symbol = :none 
    store::Bool = false 
    verbose::Bool = false                   
end 

## Multiblock

Base.@kwdef mutable struct ParMbplsda{Q <: Float}  # mbplsrda, mbplslda
    nlv::Signed = 1
    bscal::Symbol = :none   
    prior::Union{Symbol, Vector{Q}} = :prop   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParMbplsqda{Q <: Float}  
    nlv::Signed = 1
    bscal::Symbol = :none   
    prior::Union{Symbol, Vector{Q}} = :prop 
    alpha::Q = 0.   
    scal::Symbol = :none                    
end 

Base.@kwdef mutable struct ParMbplskdeda{Q <: Float}  
    nlv::Signed = 1
    bscal::Symbol = :none   
    prior::Union{Symbol, Vector{Q}} = :prop 
    h::Union{Nothing, Q, Vector{Q}} = nothing  
    a::Q = 1.  
    scal::Symbol = :none                    
end 

## Occ 

Base.@kwdef mutable struct ParOcc{Q <: Float}    # occsd, occod
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
end 

Base.@kwdef mutable struct ParOccsdod{Q <: Float}
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
    gamma::Q = .5
    fscal::Function = madv
end 

Base.@kwdef mutable struct ParOccdds{Q <: Float}
    fcentr::Function = meanv
    fscal::Function = stdv
    alpha::Q = .05 
end 

Base.@kwdef mutable struct ParOccstah{Q <: Float} 
    nlv::Signed = 500
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
    scal::Symbol = :none 
    seed::Union{Nothing, Signed} = nothing                   
end 

Base.@kwdef mutable struct ParOccknn{Q <: Float}
    nsamp::Signed = 100
    metric::Symbol = :eucl                                       
    k::Signed = 1     
    algo::Function = sum
    typcut::Symbol = :mad   
    cri::Q = 3.
    alpha::Q = .025 
    scal::Symbol = :none 
    seed::Union{Nothing, Signed} = nothing                  
end 


