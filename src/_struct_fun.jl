############---- Weights

struct Weight{T <: AbstractFloat}
    w::Vector{T} 
    #w::Union{Vector{T}, CuArray}
    #w::Union{AbstractVector}
end

############---- Data Processing 

## Preprocessing

struct DetrendLo
    par::ParDetrendLo
end

struct DetrendPol
    par::ParDetrendPol
end

struct DetrendAsls
    par::ParDetrendAsls
end

struct DetrendAirpls
    par::ParDetrendAirpls
end

struct DetrendArpls
    par::ParDetrendArpls
end

struct Fdif
    par::ParFdif
end

struct Interpl
    par::ParInterpl
end

struct Mavg
    par::ParMavg
end

struct Savgol
    par::ParSavgol
end

struct Snv
    par::ParSnv
end

struct Snorm
end

struct Center
    xmeans::Vector
end

struct Scale
    xscales::Vector
end

struct Cscale
    xmeans::Vector
    xscales::Vector
end

struct Rmgap
    par::ParRmgap
end

## Calibration transfer

struct Calds
    fitm
end

struct Calpds
    fitm
    s
end

############---- Dimension reduction

struct Pca 
    T::Matrix 
    V::Matrix
    sv::Vector
    xmeans::Vector
    xscales::Vector
    weights::Weight
    niter::Union{Vector{Int}, Nothing}    # pcanipals, pcanipalsmiss
    par::Union{ParPca, ParPcanipals, ParPcapp, ParPcaout}
end

struct Spca
    T::Matrix 
    V::Matrix
    sv::Vector
    beta::Matrix
    xmeans::Vector
    xscales::Vector
    weights::Weight
    niter::Union{Vector{Int}, Nothing}
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
    par::ParSpca
end

struct Kpca
    X::Matrix
    Kt::Adjoint
    T::Matrix
    V::Matrix
    sv::Vector  
    eig::Vector    
    DKt::Matrix
    vtot::Matrix
    xscales::Vector 
    weights::Weight
    kwargs::Base.Pairs
    par::ParKpca
end

struct Covsel
    sel::Vector{Int}
    selc::Vector
    xss::Vector
    yss::Vector
    xsstot::Real
    ysstot::Real 
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Weight
    par::ParCovsel
end

struct Rp
    T::Matrix
    V::Union{Matrix, SparseArrays.SparseMatrixCSC}
    xmeans::Vector
    xscales::Vector
    par::ParRp
end

struct Umap 
    T::Matrix
    fitm::UMAP.UMAP_
    xscales::Vector
    s::Vector{Int}
    par::ParUmap
end 
    
struct Fda
    T::Matrix
    V::Matrix
    Tcenters::Matrix
    eig::Vector
    sstot::AbstractFloat
    W::Matrix
    xmeans::Vector
    xscales::Vector
    weights::Weight
    lev::Vector
    ni::Vector{Int}
    par::ParFda
end

## Multiblock

struct Blockscal
    bscales::Vector
    xmeans::Vector{Vector}
    xscales::Vector{Vector}
    par::ParBlock
end

struct Mbconcat
    res::Nothing
end

struct Cca
    Tx::Matrix
    Ty::Matrix
    Wx::Matrix
    Wy::Matrix
    d::Vector    
    bscales::Vector    
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Weight
    par::ParCca
end

struct Ccawold
    Tx::Matrix
    Ty::Matrix
    Vx::Matrix
    Vy::Matrix
    Rx::Matrix
    Ry::Matrix    
    Wx::Matrix
    Wy::Matrix
    TTx::Vector
    TTy::Vector  
    bscales::Vector    
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Weight
    niter::Vector{Int}
    par::ParCcawold
end

struct Plscan
    Tx::Matrix
    Ty::Matrix
    Vx::Matrix
    Vy::Matrix
    Rx::Matrix
    Ry::Matrix    
    Wx::Matrix
    Wy::Matrix
    TTx::Vector
    TTy::Vector
    delta::Vector    
    bscales::Vector    
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Weight
    par::ParPls2bl
end

struct Plstuck
    Tx::Matrix
    Ty::Matrix
    Wx::Matrix
    Wy::Matrix
    TTx::Vector
    TTy::Vector
    delta::Vector
    bscales::Vector    
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Weight
    par::ParPls2bl
end

struct Rasvd
    Tx::Matrix
    Ty::Matrix
    Bx::Matrix
    Wy::Matrix
    lambda::Vector    
    bscales::Vector    
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Weight
    par::ParRasvd
end

struct Mbpca
    T::Matrix 
    U::Matrix
    W::Matrix
    Tb::Vector{Matrix}
    Tbl::Vector{Matrix}
    Vbl::Vector{Matrix}
    lb::Matrix
    mu::Vector
    fitm_bl::Blockscal
    weights::Weight
    niter::Vector{Int}
    par::ParMbpca
end

struct Comdim
    T::Matrix 
    U::Matrix
    W::Matrix
    Tb::Vector{Matrix}
    Tbl::Vector{Matrix}
    Vbl::Vector{Matrix}
    lb::Matrix
    mu::Vector
    fitm_bl::Blockscal
    weights::Weight
    niter::Vector{Int}
    par::ParMbpca
end

############---- Regression

struct Mlr
    B::Matrix   
    int::Matrix
    weights::Weight
    par::ParMlr
end

struct MlrNoArg
    B::Matrix   
    int::Matrix
    weights::Weight
end

struct Rr
    V::Matrix
    TtY::Matrix
    sv::Vector
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    weights::Weight
    par::ParRr
end

struct Rrchol
    B::Matrix   
    int::Matrix
    weights::Weight
    par::ParRr
end

struct Pcr
    fitm::Pca
    C::Matrix
    ymeans::Vector
    yscales::Vector
    par::ParPca
end

struct Plsr
    T::Matrix
    V::Matrix
    R::Matrix
    W::Matrix
    C::Matrix
    TT::Vector
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Weight
    niter::Union{Vector{Int}, Nothing}   # plswold
    par::Union{ParPlsr, ParPlswold, ParRrr}
end

struct Cglsr
    B::Matrix
    g::Vector
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    F::Union{Matrix, Nothing}
    par::ParCglsr
end

struct Spcr
    fitm::Spca
    C::Matrix
    ymeans::Vector
    yscales::Vector
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
    par::ParSpca
end

struct Splsr
    T::Matrix
    V::Matrix
    R::Matrix
    W::Matrix
    C::Matrix
    TT::Vector
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Weight
    niter::Union{Vector{Int}, Nothing}   # snipals_shen when Y with q > 1
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
    par::ParSplsr
end

struct PlsravgUnif
    fitm::Plsr
    nlv::UnitRange
end

struct Plsravg
    fitm::PlsravgUnif
    par::ParPlsr
end

struct Krr
    X::Matrix
    K::Matrix
    U::Matrix
    UtDY::Matrix
    sv::Vector
    DKt::Matrix
    vtot::Matrix
    xscales::Vector
    ymeans::Vector
    weights::Weight
    kwargs::Base.Pairs
    par::ParKrr
end

struct Kplsr
    X::Matrix
    Kt::Adjoint
    T::Matrix
    C::Matrix
    U::Matrix
    R::Matrix
    DKt::Matrix
    vtot::Matrix   
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Weight
    iter::Vector{Int}
    kwargs::Base.Pairs
    par::ParKplsr
end

struct Dkplsr
    fitm::Plsr
    X::Matrix
    K::Matrix
    xscales::Vector
    yscales::Vector
    kwargs::Base.Pairs
    par::ParKplsr
end

## Local

struct Knnr
    X::Matrix
    Y::Matrix
    xscales::Vector
    par::ParKnn
end

struct Lwmlr
    X::Matrix
    Y::Matrix
    xscales::Vector
    par::ParKnn
end

struct Lwplsr
    X::Matrix
    Y::Matrix
    fitm::Union{Nothing, Plsr}
    xscales::Vector
    par::ParLwplsr
end

struct Lwplsravg
    X::Matrix
    Y::Matrix
    fitm::Union{Nothing, Plsr}
    xscales::Vector
    par::ParLwplsr
end

struct Loessr
    fitm::Loess.LoessModel
    xscales::Vector
    par::ParLoessr
end

## Svm, Trees

struct Svmr
    fitm::LIBSVM.SVM
    xscales::Vector
    par::ParSvm
end

struct Treer
    fitm::Union{DecisionTree.Root, DecisionTree.Ensemble}
    xscales::Vector
    featur::Vector{Int}
    par::Union{ParTree, ParRf}
end

## Multiblock

struct Mbplsr
    fitm::Plsr
    T::Matrix
    R::Matrix
    C::Matrix
    fitm_bl::Blockscal
    ymeans::Vector
    yscales::Vector
    weights::Weight
    par::ParMbplsr
end

struct Mbplswest     # mbplswest, mbwcov 
    T::Matrix
    V::Matrix
    R::Matrix
    W::Matrix
    C::Matrix
    Tb::Vector{Matrix}
    Tbl::Vector{Matrix}
    Pbl::Vector{Matrix}
    TT::Vector
    fitm_bl::Blockscal
    ymeans::Vector
    yscales::Vector
    weights::Weight
    lb::Union{Matrix, Nothing}
    niter::Union{Vector, Nothing}
    par::ParMbplsr
end

struct Rosaplsr
    T::Matrix
    V::Matrix
    R::Matrix
    W::Matrix
    C::Matrix
    TT::Vector
    fitm_bl::Blockscal
    ymeans::Vector
    yscales::Vector
    weights::Weight
    bl::Vector{Int}
    par::ParSoplsr
end

struct Soplsr
    fitm::Vector
    T::Matrix
    fit::Matrix
    b::Vector
    fitm_bl::Blockscal    
    yscales::Vector
    par::ParSoplsr
end

struct Smbplsr
    fitm::Splsr
    T::Matrix
    R::Matrix
    C::Matrix
    fitm_bl::Blockscal
    ymeans::Vector
    yscales::Vector
    weights::Weight
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
    par::ParSmbplsr
end

############---- Discrimination

struct Dmnorm
    mu
    Uinv 
    detS
    cst
    par::ParDmnorm
end

struct Dmnormlog
    mu
    Uinv 
    logdetS
    logcst
    par::ParDmnorm
end

struct Dmkern
    X::Matrix
    H::Matrix
    Hinv::Matrix
    detH::Float64
    par::ParDmkern
end

struct Lda
    fitm::Vector{Dmnorm}
    W::Matrix  
    ct::Matrix
    priors::Vector
    ni::Vector{Int}
    lev::Vector
    weights::Weight
    par::ParLda
end

struct Qda
    fitm::Vector{Dmnorm}
    Wi::AbstractVector  
    ct::Matrix
    priors::Vector
    ni::Vector{Int}
    lev::Vector
    weights::Weight
    par::ParQda
end

struct Rda
    fitm::Vector{Dmnorm}
    Wi::AbstractVector  
    ct::Matrix
    priors::Vector
    ni::Vector{Int}
    lev::Vector
    xscales::Vector
    weights::Weight
    par::ParRda
end

struct Kdeda
    fitm::Vector{Dmkern}
    priors::AbstractVector
    lev::Vector
    ni::Vector{Int}
    par::ParKdeda
end

struct Mlrda
    fitm::Mlr 
    lev::Vector
    ni::Vector{Int}
    par::ParMlrda
end

struct Plsrda
    fitm::Union{Plsr, Splsr, Kplsr, Dkplsr}  
    lev::Vector
    ni::Vector{Int}
    par::Union{ParPlsda, ParSplsda, ParKplsda}
end

struct Rrda
    fitm::Union{Rr, Krr}  
    lev::Vector
    ni::Vector{Int}
    par::Union{ParRrda, ParKrrda}
end

struct Plsprobda    # plslda, plsqda, plskdeda  
    fitm_emb::Union{Plsr, Kplsr, Dkplsr, Splsr}
    fitm_da::Vector{Union{Lda, Qda, Kdeda}}  
    lev::Vector
    ni::Vector{Int}
    par::Union{ParPlsda, ParPlsqda, ParPlskdeda, ParKplsda, ParKplsqda, ParKplskdeda, 
        ParSplsda, ParSplsqda, ParSplskdeda}
end

## Local
## (from below, fitm not yet specified)

struct Knnda
    X::Matrix
    y::AbstractMatrix
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::ParKnn
end

struct Lwmlrda
    X::Matrix
    y::AbstractMatrix
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::ParKnn
end

struct Lwplsrda
    X::Matrix
    y::AbstractMatrix
    fitm
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::ParLwplsda
end

struct Lwplslda   
    X::Matrix
    y::AbstractMatrix
    fitm
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::ParLwplsda
end

struct Lwplsqda
    X::Matrix
    y::AbstractMatrix
    fitm
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::ParLwplsqda
end

## Svm, Trees

struct Svmda
    fitm
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::ParSvm
end

struct Treeda 
    fitm
    xscales::Vector
    featur::Vector{Int}
    lev::Vector
    ni::Vector{Int}
    par::Union{ParTree, ParRf}
end

## Multiblock

struct Mbplsrda
    fitm::Mbplsr  
    lev::Vector
    ni::Vector{Int}
    par::ParMbplsda
end

struct Mbplsprobda    # mbplslda, mbplsqda, mbplskdeda  
    fitm_emb::Mbplsr
    fitm_da::Vector{Union{Lda, Qda, Kdeda}}   
    lev::Vector
    ni::Vector{Int}
    par::Union{ParMbplsda, ParMbplsqda, ParMbplskdeda}
end

## Occ

struct Occsd
    d::DataFrame 
    fitm
    tscales::Vector
    e_cdf::ECDF
    cutoff::Real   
    par::ParOcc
end

struct Occod
    d::DataFrame
    fitm
    e_cdf::ECDF
    cutoff::Real   
    par::ParOcc
end

struct Occstah
    d::DataFrame
    res_stah::NamedTuple
    V::Matrix
    e_cdf::ECDF
    cutoff::Real
    par::ParOccstah
end

struct Occsdod
    d::DataFrame
    fitmsd
    fitmod
    par::ParOcc
end

struct Occknn
    d::DataFrame
    X::Matrix
    e_cdf::ECDF
    cutoff::Real
    xscales::Vector
    par::ParOccknn
end

struct Occlknn
    d::DataFrame
    X::Matrix
    e_cdf::ECDF
    cutoff::Real
    xscales::Vector
    par::ParOccknn
end

