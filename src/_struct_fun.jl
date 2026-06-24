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

struct Interpl
    par::ParInterpl
end

## Calibration transfer

struct Calds
    fitm
end

struct Calpds
    fitm
    s::Vector{Vector{Int}}
end

############---- Dimension reduction

struct Pca{Q <: AbstractFloat} 
    T::Matrix{Q} 
    V::Matrix{Q}
    sv::Vector{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    weights::ProbabilityWeights{Q}
    niter::Union{Nothing, Vector{Int}}    # pcanipals, pcanipalsmiss
    par::Union{ParPca, ParPcanipals, ParPcapp, ParPcaout}
end

struct Spca
    T::Matrix{Q}
    V::Matrix{Q}
    sv::Vector{Q}
    beta::Matrix{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    weights::ProbabilityWeights{Q}
    niter::Union{Nothing, Vector{Int}}
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
    par::ParSpca
end

struct Kpca
    X::Matrix{Q}
    Kt::Adjoint{Q}
    T::Matrix{Q}
    V::Matrix{Q}
    sv::Vector{Q}  
    eig::Vector{Q}    
    DKt::Matrix{Q}
    vtot::Matrix{Q}
    xscales::Vector{Q} 
    weights::ProbabilityWeights{Q}
    kwargs::Base.Pairs
    par::ParKpca
end

struct Covsel
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
    weights::ProbabilityWeights{Q}
    par::ParCovsel
end

struct Rp
    T::Matrix{Q}
    V::Union{Matrix, SparseArrays.SparseMatrixCSC}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    par::ParRp
end

struct Umap 
    fitm::UMAP.UMAPResult    
    T::Matrix{Q}
    xscales::Vector{Q}
    s::Vector{Int}
    par::ParUmap
end 
    
struct Fda
    T::Matrix{Q}
    V::Matrix{Q}
    Tcenters::Matrix{Q}
    eig::Vector{Q}
    sstot::Q
    W::Matrix{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    weights::ProbabilityWeights{Q}
    par::ParFda
end

## Multiblock

struct Blockscal{Q <: AbstractFloat}
    bscales::Vector{Q}
    xmeans::Vector{Vector{Q}}
    xscales::Vector{Vector{Q}}
    par::ParBlock
end

struct MbconcatVector{Vector{Q}}
    res::Nothing
end

struct Cca
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
    weights::ProbabilityWeights{Q}
    par::ParCca
end

struct Ccawold
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
    weights::ProbabilityWeights{Q}
    niter::Vector{Int}
    par::ParCcawold
end

struct Plscan
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
    weights::ProbabilityWeights{Q}
    par::ParPls2bl
end

struct Plstuck
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
    weights::ProbabilityWeights{Q}
    par::ParPls2bl
end

struct Rasvd
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
    weights::ProbabilityWeights{Q}
    par::ParRasvd
end

struct Cpca
    T::Matrix{Q}
    U::Matrix{Q}
    W::Matrix{Q}
    Tb::Vector{Matrix{Q}}
    Tbl::Vector{Matrix{Q}}
    Vbl::Vector{Matrix{Q}}
    lb::Matrix{Q}
    mu::Vector{Q}
    fitm_bl::Blockscal
    weights::ProbabilityWeights{Q}
    niter::Vector{Int}
    par::ParCpca
end

struct Comdim
    T::Matrix{Q}
    U::Matrix{Q}
    W::Matrix{Q}
    Tb::Vector{Matrix{Q}}
    Tbl::Vector{Matrix{Q}}
    Vbl::Vector{Matrix{Q}}
    lb::Matrix{Q}
    mu::Vector{Q}
    fitm_bl::Blockscal
    weights::ProbabilityWeights{Q}
    niter::Vector{Int}
    par::ParCpca
end

############---- Regression

struct Mlr
    B::Matrix{Q}   
    int::Matrix{Q}
    weights::ProbabilityWeights{Q}
    par::ParMlr
end

struct Mlrnoarg
    B::Matrix{Q}   
    int::Matrix{Q}
    weights::ProbabilityWeights{Q}
end

struct Decompx
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

struct Rr
    V::Matrix{Q}
    TtY::Matrix{Q}
    sv::Vector{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    weights::ProbabilityWeights{Q}
    par::ParRr
end

struct Rrchol
    B::Matrix{Q}   
    int::Matrix{Q}
    weights::ProbabilityWeights{Q}
    par::ParRr
end

struct Pcr
    fitm::Pca
    C::Matrix{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    par::ParPca
end

struct Plsr
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
    weights::ProbabilityWeights{Q}
    niter::Union{Nothing, Vector{Int}}   # for plswold
    par::Union{ParPlsr, ParPlswold, ParRrr}
end

struct Plsravgunif
    fitm::Plsr
    par::ParPlsravgunif
end

struct Plsravg
    fitm::Plsravgunif
    par::ParPlsravg
end

struct Cglsr
    B::Matrix{Q}
    g::Vector{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    F::Union{Nothing, Matrix{Q}}
    par::ParCglsr
end

struct Spcr
    fitm::Spca
    C::Matrix{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    par::ParSpca
end

struct Splsr
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
    weights::ProbabilityWeights{Q}
    niter::Union{Nothing, Vector{Int}}   # snipals_shen when Y with q > 1
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
    par::ParSplsr
end

struct Krr
    X::Matrix{Q}
    K::Matrix{Q}
    U::Matrix{Q}
    UtDY::Matrix{Q}
    sv::Vector{Q}
    DKt::Matrix{Q}
    vtot::Matrix{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    weights::ProbabilityWeights{Q}
    kwargs::Base.Pairs
    par::ParKrr
end

struct Kplsr
    X::Matrix{Q}
    Kt::Adjoint{Q}
    T::Matrix{Q}
    C::Matrix{Q}
    U::Matrix{Q}
    R::Matrix{Q}
    DKt::Matrix{Q}
    vtot::Matrix{Q}   
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights{Q}
    iter::Vector{Int}
    kwargs::Base.Pairs
    par::ParKplsr
end

struct Dkplsr
    fitm::Plsr
    X::Matrix{Q}
    K::Matrix{Q}
    xscales::Vector{Q}
    yscales::Vector{Q}
    kwargs::Base.Pairs
    par::ParKplsr
end

## Local

struct Knnr
    X::Matrix{Q}
    Y::Matrix{Q}
    xscales::Vector{Q}
    par::ParKnn
end

struct Lwmlr
    X::Matrix{Q}
    Y::Matrix{Q}
    xscales::Vector{Q}
    par::ParLwmlr
end

struct Lwplsr
    fitm::Union{Nothing, Plsr}
    X::Matrix{Q}
    Y::Matrix{Q}
    xscales::Vector{Q}
    par::ParLwplsr
end

struct Lwplsravg
    fitm::Union{Nothing, Plsr}
    X::Matrix{Q}
    Y::Matrix{Q}
    xscales::Vector{Q}
    par::ParLwplsravg
end

struct Loessr
    fitm::Loess.LoessModel
    xscales::Vector{Q}
    par::ParLoessr
end

## Svm, Trees

struct Svmr
    fitm::LIBSVM.SVM
    xscales::Vector{Q}
    par::ParSvm
end

struct Treer
    fitm::Union{DecisionTree.Root, DecisionTree.Ensemble}
    xscales::Vector{Q}
    featur::Vector{Int}
    par::Union{ParTree, ParRf}
end

## Multiblock

struct Mbplsr
    fitm_bl::Blockscal
    fitm::Plsr
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights{Q}
    par::ParMbplsr
end

struct Soplsr
    fitm_bl::Blockscal    
    fitm::Vector{Q}
    T::Matrix{Q}
    fit::Matrix{Q}
    b::Vector{Q}
    yscales::Vector{Q}
    par::ParSoplsr
end

struct Rosaplsr
    fitm_bl::Blockscal
    T::Matrix{Q}
    V::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    TT::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights{Q}
    bl::Vector{Int}
    par::ParRosaplsr
end

struct Mbplswest     # mbplswest, mbwcov 
    T::Matrix{Q}
    V::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    Tb::Vector{Matrix{Q}}
    Tbl::Vector{Matrix{Q}}
    Vbl::Vector{Matrix{Q}}
    TT::Vector{Q}
    fitm_bl::Blockscal
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::ProbabilityWeights{Q}
    lb::Union{Nothing, Matrix{Q}}
    niter::Union{Vector{Int}, Nothing}
    par::ParMbplsr
end

############---- Discrimination

struct Dmnorm
    mu::Vector{Q}
    Uinv::Matrix{Q} 
    detS::Q
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
    X::Matrix{Q}
    H::Matrix{Q}
    Hinv::Matrix{Q}
    detH::Q
    par::ParDmkern
end

struct Lda
    fitm::Vector{Dmnorm}
    W::Matrix{Q}  
    ct::Matrix{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    weights::ProbabilityWeights{Q}
    par::ParLda
end

struct Qda
    fitm::Vector{Dmnorm}
    Wi::AbstractVector{Q}  
    ct::Matrix{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    weights::ProbabilityWeights{Q}
    par::ParQda
end

struct Rda
    fitm::Vector{Dmnorm}
    Wi::AbstractVector{Q}  
    ct::Matrix{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    xscales::Vector{Q}
    weights::ProbabilityWeights{Q}
    par::ParRda
end

struct Kdeda
    fitm::Vector{Dmkern}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::ParKdeda
end

struct Mlrda
    fitm_emb::Mlr 
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::ParMlrda
end

struct Rrda
    fitm_emb::Union{Rr, Krr}  
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::Union{ParRrda, ParKrrda}
end

struct Plsrda
    fitm_emb::Union{Plsr, Splsr, Kplsr, Dkplsr} 
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::Union{ParPlsda, ParSplsda, ParKplsda}
end

struct Plsprobda    # plslda, plsqda, plskdeda  
    fitm_emb::Union{Plsr, Splsr, Kplsr, Dkplsr}
    fitm_da::Vector{Union{Lda, Qda, Kdeda}}  
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::Union{ParPlsda, ParPlsqda, ParPlskdeda, ParSplsda, ParSplsqda, ParSplskdeda,
        ParKplsda, ParKplsqda, ParKplskdeda}
end

## Local
## (from below, fitm not yet specified)

struct Knnda
    X::Matrix{Q}
    y::AbstractMatrix
    xscales::Vector{Q}
    ni::Vector{Int}
    lev::Vector{String}
    par::ParKnn
end

struct Lwmlrda
    X::Matrix{Q}
    y::AbstractMatrix
    xscales::Vector{Q}
    ni::Vector{Int}
    lev::Vector{String}
    par::ParLwmlr
end

struct Lwplsrda
    fitm
    X::Matrix{Q}
    y::AbstractMatrix
    xscales::Vector{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::ParLwplsda
end

struct Lwplslda   
    fitm
    X::Matrix{Q}
    y::AbstractMatrix
    xscales::Vector{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::ParLwplsda
end

struct Lwplsqda
    fitm
    X::Matrix{Q}
    y::AbstractMatrix
    xscales::Vector{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::ParLwplsqda
end

## Svm, Trees

struct Svmda
    fitm
    xscales::Vector{Q}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::ParSvm
end

struct Treeda 
    fitm
    xscales::Vector{Q}
    featur::Vector{Int}
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::Union{ParTree, ParRf}
end

## Multiblock

struct Mbplsrda
    fitm_emb::Mbplsr 
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::ParMbplsda
end

struct Mbplsprobda    # mbplslda, mbplsqda, mbplskdeda  
    fitm_emb::Mbplsr
    fitm_da::Vector{Union{Lda, Qda, Kdeda}}   
    ni::Vector{Int}
    priors::Vector{Q}
    lev::Vector{String}
    par::Union{ParMbplsda, ParMbplsqda, ParMbplskdeda}
end

## Occ

struct Occsd
    d::DataFrame 
    fitm
    tscales::Vector{Q}
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
    V::Matrix{Q}
    e_cdf::ECDF
    cutoff::Real
    par::ParOccstah
end

struct Occsdod
    d::DataFrame
    fitm
    e_cdf::ECDF
    cutoff::Real
    sd::NamedTuple   
    od::NamedTuple   
    sdod::NamedTuple
    coefs::Vector{Q}
    par::ParOccsdod
end

struct Occdds
    d::DataFrame
    fitm
    e_cdf::ECDF
    nu::Int
    cutoff::Real
    sd2::NamedTuple   
    od2::NamedTuple   
    coefs::Vector{Q}
    par::ParOccdds
end

struct Occknn
    d::DataFrame
    X::Matrix{Q}
    e_cdf::ECDF
    cutoff::Real
    xscales::Vector{Q}
    par::ParOccknn
end

struct Occlknn
    d::DataFrame
    X::Matrix{Q}
    e_cdf::ECDF
    cutoff::Real
    xscales::Vector{Q}
    par::ParOccknn
end
