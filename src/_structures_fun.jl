############---- Weights

struct Weight{T <: AbstractFloat}
    w::Vector{T} 
    #w::Union{Vector{T}, CuArray}
    #w::Union{AbstractVector}
end

############---- Data Processing 

## Preprocessing


struct Detrend
    par::ParDetrend
end

struct Asls
    par::ParAsls
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
    fm
end

struct Calpds
    fm
    s
end

############---- Dimension reduction

struct Pca 
    T::Matrix 
    P::Matrix
    sv::Vector
    xmeans::Vector
    xscales::Vector
    weights::Weight
    niter::Union{Vector{Int}, Nothing}    # pcanipals, pcanipalsmiss
    par::Union{ParPca, ParPcanipals, ParPcapp, ParPcaout}
end

struct Spca
    T::Matrix 
    P::Matrix
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
    P::Matrix
    sv::Vector  
    eig::Vector    
    D::Diagonal
    DKt::Matrix
    vtot::Matrix
    xscales::Vector 
    weights::Weight
    kwargs::Base.Pairs
    par::ParKpca
end

struct Rp
    T::Matrix
    P::Union{Matrix, SparseArrays.SparseMatrixCSC}
    xmeans::Vector
    xscales::Vector
    par::ParRp
end

struct Umap 
    T::Matrix
    fm::UMAP.UMAP_
    xscales::Vector
    par::ParUmap
end 
    
struct Fda
    T::Matrix
    P::Matrix
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
    Px::Matrix
    Py::Matrix
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
    Px::Matrix
    Py::Matrix
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
    Tbl::Vector{Matrix}
    Tb::Vector{Matrix}
    Wbl::Vector{Matrix}
    lb::Matrix
    mu::Vector
    fmbl::Blockscal
    weights::Weight
    niter::Vector{Int}
    par::ParMbpca
end

struct Comdim
    T::Matrix 
    U::Matrix
    W::Matrix
    Tbl::Vector{Matrix}
    Tb::Vector{Matrix}
    Wbl::Vector{Matrix}
    lb::Matrix
    mu::Vector
    fmbl::Blockscal
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

struct Plsr
    T::Matrix
    P::Matrix
    R::Matrix
    W::Matrix
    C::Matrix
    TT::Vector
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Weight
    niter::Union{Vector{Int}, Nothing}
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

struct Pcr
    fmpca::Pca
    T::Matrix
    R::Matrix
    C::Matrix
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Weight
    par::ParPcr
end

struct Splsr
    T::Matrix
    P::Matrix
    R::Matrix
    W::Matrix
    C::Matrix
    TT::Vector
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Weight
    niter::Union{Matrix, Nothing}
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
    par::ParSplsr
end

struct PlsravgUnif
    fm::Plsr
    nlv::UnitRange
end

struct Plsravg
    fm::PlsravgUnif
    par::ParPlsr
end

struct Kplsr
    X::Matrix
    Kt::Adjoint
    T::Matrix
    C::Matrix
    U::Matrix
    R::Matrix
    D::Diagonal 
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
    X::Matrix
    fm::Plsr
    K::Matrix
    T::Matrix
    xscales::Vector
    yscales::Vector
    kwargs::Base.Pairs
    par::ParKplsr
end

struct Rr
    V::Matrix
    TtDY::Matrix
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

struct Krr
    X::Matrix
    K::Matrix
    U::Matrix
    UtDY::Matrix
    sv::Vector
    D::Diagonal
    sqrtD::Diagonal
    DKt::Matrix
    vtot::Matrix
    xscales::Vector
    ymeans::Vector
    weights::Weight
    kwargs::Base.Pairs
    par::ParKrr
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
    fm::Union{Nothing, Plsr}
    xscales::Vector
    par::ParLwplsr
end

struct Lwplsravg
    X::Matrix
    Y::Matrix
    fm::Union{Nothing, Plsr}
    xscales::Vector
    par::ParLwplsr
end

## Svm, Trees

struct Svmr
    fm::LIBSVM.SVM
    xscales::Vector
    par::ParSvm
end

struct Treer
    fm::Union{DecisionTree.Root, DecisionTree.Ensemble}
    xscales::Vector
    featur::Vector{Int}
    par::Union{ParTree, ParRf}
end

## Multiblock

struct Mbplsr
    fm::Plsr
    T::Matrix
    R::Matrix
    C::Matrix
    fmbl::Blockscal
    ymeans::Vector
    yscales::Vector
    weights::Weight
    par::ParMbplsr
end

struct Mbplswest     # mbplswest, mbwcov 
    T::Matrix
    P::Matrix
    R::Matrix
    W::Matrix
    C::Matrix
    Tbl::Vector{Matrix}
    Tb::Vector{Matrix}
    Pbl::Vector{Matrix}
    TT::Vector
    fmbl::Blockscal
    ymeans::Vector
    yscales::Vector
    weights::Weight
    lb::Union{Matrix, Nothing}
    niter::Union{Vector, Nothing}
    par::ParMbplsr
end

struct Rosaplsr
    T::Matrix
    P::Matrix
    R::Matrix
    W::Matrix
    C::Matrix
    TT::Vector
    fmbl::Blockscal
    ymeans::Vector
    yscales::Vector
    weights::Weight
    bl::Vector{Int}
    par::ParSoplsr
end

struct Soplsr
    fm::Vector
    T::Matrix
    fit::Matrix
    b::Vector
    fmbl::Blockscal    
    yscales::Vector
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

struct Dmkern
    X::Matrix
    H::Matrix
    Hinv::Matrix
    detH::Float64
    par::ParDmkern
end

struct Lda
    fm::Vector{Dmnorm}
    W::Matrix  
    ct::Matrix
    priors::Vector
    ni::Vector{Int}
    lev::Vector
    weights::Weight
    par::ParLda
end

struct Qda
    fm::Vector{Dmnorm}
    Wi::AbstractVector  
    ct::Matrix
    priors::Vector
    ni::Vector{Int}
    lev::Vector
    weights::Weight
    par::ParQda
end

struct Rda
    fm::Vector{Dmnorm}
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
    fm::Vector{Dmkern}
    priors::AbstractVector
    lev::Vector
    ni::Vector{Int}
    par::ParKdeda
end

struct Mlrda
    fm::Mlr 
    lev::Vector
    ni::Vector{Int}
    par::ParMlrda
end

struct Plsrda
    fm::Union{Plsr, Splsr, Kplsr, Dkplsr}  
    lev::Vector
    ni::Vector{Int}
    par::Union{ParPlsda, ParSplsda, ParKplsda}
end

struct Rrda
    fm::Union{Rr, Krr}  
    lev::Vector
    ni::Vector{Int}
    par::Union{ParRrda, ParKrrda}
end

struct Plsprobda    # plslda, plsqda, plskdeda  
    fm::NamedTuple  
    lev::Vector
    ni::Vector{Int}
    par::Union{ParPlsda, ParPlsqda, ParPlskdeda, ParSplsda, ParSplsqda, ParSplskdeda,
        ParKplsda, ParKplsqda, ParKplskdeda}
end

## Local
## (from below, fm not yet specified)

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
    fm
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::ParLwplsda
end

struct Lwplslda   
    X::Matrix
    y::AbstractMatrix
    fm
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::ParLwplsda
end

struct Lwplsqda
    X::Matrix
    y::AbstractMatrix
    fm
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::ParLwplsqda
end

## Svm, Trees

struct Svmda
    fm
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::ParSvm
end

struct Treeda 
    fm
    xscales::Vector
    featur::Vector{Int}
    lev::Vector
    ni::Vector{Int}
    par::Union{ParTree, ParRf}
end

## Multiblock

struct Mbplsrda
    fm::Mbplsr  
    lev::Vector
    ni::Vector{Int}
    par::ParMbplsda
end

struct Mbplsprobda    # mbplslda, mbplsqda, mbplskdeda  
    fm::NamedTuple  
    lev::Vector
    ni::Vector{Int}
    par::Union{ParMbplsda, ParMbplsqda, ParMbplskdeda}
end

## Occ

struct Occsd
    d::DataFrame 
    fm
    tscales::Vector
    e_cdf::ECDF
    cutoff::Real   
    par::ParOcc
end

struct Occod
    d::DataFrame
    fm
    e_cdf::ECDF
    cutoff::Real   
    par::ParOcc
end

struct Occsdod
    d::DataFrame
    fmsd
    fmod
    par::ParOcc
end

struct Occstah
    d::DataFrame
    res_stah::NamedTuple
    P::Matrix
    e_cdf::ECDF
    cutoff::Real
    par::ParOccstah
end


