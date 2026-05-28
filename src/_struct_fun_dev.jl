############---- Data Processing 

## Preprocessing

struct Detrendlo
    par::ParDetrendlo
end

struct Detrendpol
    par::ParDetrendpol
end

struct Detrendasls
    par::ParDetrendasls
end

struct Detrendairpls
    par::ParDetrendairpls
end

struct Detrendarpls
    par::ParDetrendarpls
end

struct Emsc{Q <: AbstractFloat}
    xref::Vector{Q}
    Xr::Matrix{Q}
    par::ParEmsc
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

struct Center{Q <: AbstractFloat}
    xmeans::Vector{Q}
end

struct Scale{Q <: AbstractFloat}
    xscales::Vector{Q}
end

struct Cscale{Q <: AbstractFloat}
    xmeans::Vector{Q}
    xscales::Vector{Q}
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

struct Pca{Q <: AbstractFloat}
    T::Matrix{Q} 
    V::Matrix{Q}
    sv::Vector{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    weights::ProbabilityWeights
    niter::Union{Vector{Int}, Nothing}    # pcanipals, pcanipalsmiss
    par::Union{ParPca, ParPcanipals, ParPcapp, ParPcaout}
end

struct Spca{Q <: AbstractFloat}
    T::Matrix{Q} 
    V::Matrix{Q}
    sv::Vector{Q}
    beta::Matrix{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    weights::ProbabilityWeights
    niter::Union{Vector{Int}, Nothing}
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
    par::ParSpca
end

struct Kpca{Q <: AbstractFloat}
    X::Matrix{Q}
    Kt::Adjoint
    T::Matrix{Q}
    V::Matrix{Q}
    sv::Vector{Q} 
    eig::Vector{Q}   
    DKt::Matrix{Q}
    vtot::Matrix{Q}
    xscales::Vector{Q}
    weights::ProbabilityWeights
    kwargs::Base.Pairs
    par::ParKpca
end

struct Covsel{Q <: AbstractFloat}
    sel::Vector{Int}
    selc::Vector{Q}
    xss::Vector{Q}
    yss::Vector{Q}
    xsstot::Q
    ysstot::Q 
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights
    par::ParCovsel
end

struct Rp{Q <: AbstractFloat}
    T::Matrix{Q}
    V::Union{Matrix, SparseArrays.SparseMatrixCSC}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    par::ParRp
end

struct Umap{Q <: AbstractFloat} 
    fitm::UMAP.UMAPResult    
    T::Matrix{Q}
    xscales::Vector{Q}
    s::Vector{Int}
    par::ParUmap
end 
    
struct Fda{Q <: AbstractFloat}
    T::Matrix{Q}
    V::Matrix{Q}
    Tcenters::Matrix{Q}
    eig::Vector{Q}
    sstot::Q
    W::Matrix{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    weights::ProbabilityWeights
    par::ParFda
end

## Multiblock

struct Blockscal{Q <: AbstractFloat}
    bscales::Vector{Q}
    xmeans::Vector{Vector{Q}}
    xscales::Vector{Vector{Q}}
    par::ParBlock
end

struct Mbconcat
    res::Nothing
end

struct Cca{Q <: AbstractFloat}
    Tx::Matrix{Q}
    Ty::Matrix{Q}
    Wx::Matrix{Q}
    Wy::Matrix{Q}
    d::Vector{Q}   
    bscales::Vector{Q}   
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights
    par::ParCca
end

struct Ccawold{Q <: AbstractFloat}
    Tx::Matrix{Q}
    Ty::Matrix{Q}
    Vx::Matrix{Q}
    Vy::Matrix{Q}
    Rx::Matrix{Q}
    Ry::Matrix{Q}    
    Wx::Matrix{Q}
    Wy::Matrix{Q}
    TTx::Vector{Q}
    TTy::Vector{Q} 
    bscales::Vector{Q}   
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights
    niter::Vector{Int}
    par::ParCcawold
end

struct Plscan{Q <: AbstractFloat}
    Tx::Matrix{Q}
    Ty::Matrix{Q}
    Vx::Matrix{Q}
    Vy::Matrix{Q}
    Rx::Matrix{Q}
    Ry::Matrix{Q}    
    Wx::Matrix{Q}
    Wy::Matrix{Q}
    TTx::Vector{Q}
    TTy::Vector{Q}
    delta::Vector{Q}   
    bscales::Vector{Q}   
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights
    par::ParPls2bl
end

struct Plstuck{Q <: AbstractFloat}
    Tx::Matrix{Q}
    Ty::Matrix{Q}
    Wx::Matrix{Q}
    Wy::Matrix{Q}
    TTx::Vector{Q}
    TTy::Vector{Q}
    delta::Vector{Q}
    bscales::Vector{Q}   
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights
    par::ParPls2bl
end

struct Rasvd{Q <: AbstractFloat}
    Tx::Matrix{Q}
    Ty::Matrix{Q}
    Bx::Matrix{Q}
    Wy::Matrix{Q}
    lambda::Vector{Q}   
    bscales::Vector{Q}   
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights
    par::ParRasvd
end

struct Mbpca{Q <: AbstractFloat}
    T::Matrix{Q} 
    U::Matrix{Q}
    W::Matrix{Q}
    Tb::Vector{Matrix}
    Tbl::Vector{Matrix}
    Vbl::Vector{Matrix}
    lb::Matrix{Q}
    mu::Vector{Q}
    fitm_bl::Blockscal
    weights::ProbabilityWeights
    niter::Vector{Int}
    par::ParMbpca
end

struct Comdim{Q <: AbstractFloat}
    T::Matrix{Q} 
    U::Matrix{Q}
    W::Matrix{Q}
    Tb::Vector{Matrix}
    Tbl::Vector{Matrix}
    Vbl::Vector{Matrix}
    lb::Matrix{Q}
    mu::Vector{Q}
    fitm_bl::Blockscal
    weights::ProbabilityWeights
    niter::Vector{Int}
    par::ParMbpca
end

############---- Regression

struct Mlr{Q <: AbstractFloat}
    B::Matrix{Q}   
    int::Matrix{Q}
    weights::ProbabilityWeights
    par::ParMlr
end

struct Mlrnoarg{Q <: AbstractFloat}
    B::Matrix{Q}   
    int::Matrix{Q}
    weights::ProbabilityWeights
end

struct Decompx{Q <: AbstractFloat}
    fit::NamedTuple
    R::Matrix{Q}
    mat::NamedTuple
    ss::NamedTuple
    df::NamedTuple
    f::StatsModels.FormulaTerm
    assign::Vector{Int}
    dat::DataFrame
    xmeans::Vector{Q}
end

struct Rr{Q <: AbstractFloat}
    V::Matrix{Q}
    TtY::Matrix{Q}
    sv::Vector{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    weights::ProbabilityWeights
    par::ParRr
end

struct Rrchol{Q <: AbstractFloat}
    B::Matrix{Q}   
    int::Matrix{Q}
    weights::ProbabilityWeights
    par::ParRr
end

struct Pcr{Q <: AbstractFloat}
    fitm::Pca
    C::Matrix{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    par::ParPca
end

struct Plsr{Q <: AbstractFloat}
    T::Matrix{Q}
    V::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    TT::Vector{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights
    niter::Union{Vector{Int}, Nothing}   # plswold
    par::Union{ParPlsr, ParPlswold, ParRrr}
end

struct Cglsr{Q <: AbstractFloat}
    B::Matrix{Q}
    g::Vector{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    F::Union{Matrix, Nothing}
    par::ParCglsr
end

struct Spcr{Q <: AbstractFloat}
    fitm::Spca
    C::Matrix{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    par::ParSpca
end

struct PlsravgUnif
    fitm::Plsr
    nlv::UnitRange
end

struct Plsravg
    fitm::PlsravgUnif
    par::ParPlsr
end

struct Splsr{Q <: AbstractFloat}
    T::Matrix{Q}
    V::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    TT::Vector{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights
    niter::Union{Vector{Int}, Nothing}   # snipals_shen when Y with q > 1
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
    par::ParSplsr
end

struct Krr{Q <: AbstractFloat}
    X::Matrix{Q}
    K::Matrix{Q}
    U::Matrix{Q}
    UtDY::Matrix{Q}
    sv::Vector{Q}
    DKt::Matrix{Q}
    vtot::Matrix{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    weights::ProbabilityWeights
    kwargs::Base.Pairs
    par::ParKrr
end

struct Kplsr{Q <: AbstractFloat}
    X::Matrix{Q}
    Kt::Adjoint
    T::Matrix{Q}
    C::Matrix{Q}
    U::Matrix{Q}
    R::Matrix{Q}
    DKt::Matrix{Q}
    vtot::Matrix{Q}   
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights
    iter::Vector{Int}
    kwargs::Base.Pairs
    par::ParKplsr
end

struct Dkplsr{Q <: AbstractFloat}
    fitm::Plsr
    X::Matrix{Q}
    K::Matrix{Q}
    xscales::Vector{Q}
    yscales::Vector{Q}
    kwargs::Base.Pairs
    par::ParKplsr
end

## Local

struct Knnr{Q <: AbstractFloat}
    X::Matrix{Q}
    Y::Matrix{Q}
    xscales::Vector{Q}
    par::ParKnn
end

struct Lwmlr{Q <: AbstractFloat}
    X::Matrix{Q}
    Y::Matrix{Q}
    xscales::Vector{Q}
    par::ParLwmlr
end

struct Lwplsr{Q <: AbstractFloat}
    fitm::Union{Nothing, Plsr}
    X::Matrix{Q}
    Y::Matrix{Q}
    xscales::Vector{Q}
    par::ParLwplsr
end

struct Lwplsravg{Q <: AbstractFloat}
    fitm::Union{Nothing, Plsr}
    X::Matrix{Q}
    Y::Matrix{Q}
    xscales::Vector{Q}
    par::ParLwplsr
end

struct Loessr{Q <: AbstractFloat}
    fitm::Loess.LoessModel
    xscales::Vector{Q}
    par::ParLoessr
end

## Svm, Trees

struct Svmr{Q <: AbstractFloat}
    fitm::LIBSVM.SVM
    xscales::Vector{Q}
    par::ParSvm
end

struct Treer{Q <: AbstractFloat}
    fitm::Union{DecisionTree.Root, DecisionTree.Ensemble}
    xscales::Vector{Q}
    featur::Vector{Int}
    par::Union{ParTree, ParRf}
end

## Multiblock

struct Mbplsr{Q <: AbstractFloat}
    fitm_bl::Blockscal
    fitm::Plsr
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights
    par::ParMbplsr
end

struct Mbplswest{Q <: AbstractFloat}     # mbplswest, mbwcov 
    T::Matrix{Q}
    V::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    Tb::Vector{Matrix}
    Tbl::Vector{Matrix}
    Pbl::Vector{Matrix}
    TT::Vector{Q}
    fitm_bl::Blockscal
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights
    lb::Union{Matrix, Nothing}
    niter::Union{Vector, Nothing}
    par::ParMbplsr
end

struct Rosaplsr{Q <: AbstractFloat}
    fitm_bl::Blockscal
    T::Matrix{Q}
    V::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    TT::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights
    bl::Vector{Int}
    par::ParSoplsr
end

struct Soplsr{Q <: AbstractFloat}
    fitm_bl::Blockscal    
    fitm::Vector{Q}
    T::Matrix{Q}
    fit::Matrix{Q}
    b::Vector{Q}
    yscales::Vector{Q}
    par::ParSoplsr
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

struct Dmkern{Q <: AbstractFloat}
    X::Matrix{Q}
    H::Matrix{Q}
    Hinv::Matrix{Q}
    detH::Q
    par::ParDmkern
end

struct Lda{Q <: AbstractFloat}
    fitm::Vector{Dmnorm}
    W::Matrix{Q}  
    ct::Matrix{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    weights::ProbabilityWeights
    par::ParLda
end

struct Qda{Q <: AbstractFloat}
    fitm::Vector{Dmnorm}
    Wi::AbstractVector  
    ct::Matrix{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    weights::ProbabilityWeights
    par::ParQda
end

struct Rda{Q <: AbstractFloat}
    fitm::Vector{Dmnorm}
    Wi::AbstractVector  
    ct::Matrix{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    xscales::Vector{Q}
    weights::ProbabilityWeights
    par::ParRda
end

struct Kdeda{Q <: AbstractFloat}
    fitm::Vector{Dmkern}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    par::ParKdeda
end

struct Mlrda{Q <: AbstractFloat}
    fitm::Mlr 
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    par::ParMlrda
end

struct Rrda{Q <: AbstractFloat}
    fitm::Union{Rr, Krr}  
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    par::Union{ParRrda, ParKrrda}
end

struct Plsrda{Q <: AbstractFloat}
    fitm::Union{Plsr, Splsr, Kplsr, Dkplsr} 
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    par::Union{ParPlsda, ParSplsda, ParKplsda}
end

struct Plsprobda{Q <: AbstractFloat}    # plslda, plsqda, plskdeda  
    fitm_emb::Union{Plsr, Splsr, Kplsr, Dkplsr}
    fitm_da::Vector{Union{Lda, Qda, Kdeda}}  
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    par::Union{ParPlsda, ParPlsqda, ParPlskdeda, ParSplsda, ParSplsqda, ParSplskdeda,
        ParKplsda, ParKplsqda, ParKplskdeda}
end

## Local
## (from below, fitm not yet specified)

struct Knnda{Q <: AbstractFloat}
    X::Matrix{Q}
    y::AbstractMatrix
    xscales::Vector{Q}
    ni::Vector{Int}
    lev::Vector{Q}
    par::ParKnn
end

struct Lwmlrda{Q <: AbstractFloat}
    X::Matrix{Q}
    y::AbstractMatrix
    xscales::Vector{Q}
    ni::Vector{Int}
    lev::Vector{Q}
    par::ParLwmlr
end

struct Lwplsrda{Q <: AbstractFloat}
    fitm
    X::Matrix{Q}
    y::AbstractMatrix
    xscales::Vector{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    par::ParLwplsda
end

struct Lwplslda{Q <: AbstractFloat}
    fitm
    X::Matrix{Q}
    y::AbstractMatrix
    xscales::Vector{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    par::ParLwplsda
end

struct Lwplsqda{Q <: AbstractFloat}
    fitm
    X::Matrix{Q}
    y::AbstractMatrix
    xscales::Vector{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    par::ParLwplsqda
end

## Svm, Trees

struct Svmda{Q <: AbstractFloat}
    fitm
    xscales::Vector{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    par::ParSvm
end

struct Treeda{Q <: AbstractFloat}
    fitm
    xscales::Vector{Q}
    featur::Vector{Int}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    par::Union{ParTree, ParRf}
end

## Multiblock

struct Mbplsrda{Q <: AbstractFloat}
    fitm::Mbplsr 
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    par::ParMbplsda
end

struct Mbplsprobda{Q <: AbstractFloat}    # mbplslda, mbplsqda, mbplskdeda  
    fitm_emb::Mbplsr
    fitm_da::Vector{Union{Lda, Qda, Kdeda}}   
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{Q}
    par::Union{ParMbplsda, ParMbplsqda, ParMbplskdeda}
end

## Occ

struct Occsd{Q <: AbstractFloat}
    d::DataFrame 
    fitm
    tscales::Vector{Q}
    e_cdf::ECDF
    cutoff::Q   
    par::ParOcc
end

struct Occod{Q <: AbstractFloat}
    d::DataFrame
    fitm
    e_cdf::ECDF
    cutoff::Q   
    par::ParOcc
end

struct Occstah{Q <: AbstractFloat}
    d::DataFrame
    res_stah::NamedTuple
    V::Matrix{Q}
    e_cdf::ECDF
    cutoff::Q
    par::ParOccstah
end

struct Occsdod{Q <: AbstractFloat}
    d::DataFrame
    fitm
    e_cdf::ECDF
    cutoff::Q
    sd::NamedTuple   
    od::NamedTuple   
    sdod::NamedTuple
    coefs::Vector{Q}
    par::ParOccsdod
end

struct Occdds{Q <: AbstractFloat}
    d::DataFrame
    fitm
    e_cdf::ECDF
    nu::Int
    cutoff::Q
    sd2::NamedTuple   
    od2::NamedTuple   
    coefs::Vector{Q}
    par::ParOccdds
end

struct Occknn{Q <: AbstractFloat}
    d::DataFrame
    X::Matrix{Q}
    e_cdf::ECDF
    cutoff::Q
    xscales::Vector{Q}
    par::ParOccknn
end

struct Occlknn{Q <: AbstractFloat}
    d::DataFrame
    X::Matrix{Q}
    e_cdf::ECDF
    cutoff::Q
    xscales::Vector{Q}
    par::ParOccknn
end

