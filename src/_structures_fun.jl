######---- Weights

struct Weight{T <: AbstractFloat}
    w::Vector{T} 
end

######---- Dimension reduction

struct Pca 
    T::Matrix 
    P::Matrix
    sv::Vector
    xmeans::Vector
    xscales::Vector
    weights::Weight
    niter::Union{Vector{Int}, Nothing} # for PCA Nipals
    par::Par
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
    par::Par
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
    par::Par
end

struct Rp
    T::Matrix
    P::Union{Matrix, SparseArrays.SparseMatrixCSC}
    xmeans::Vector
    xscales::Vector
    par::Par
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
    par::Par
end

## Multiblock

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
    par::Par
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
    par::Par
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
    par::Par
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
    par::Par
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
    par::Par
end

struct Blockscal
    bscales::Vector
    xmeans::Vector{Vector}
    xscales::Vector{Vector}
    par::ParBlock
end

struct Mbconcat
    res::Nothing
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
    fmsc::Blockscal
    weights::Weight
    niter::Vector{Int}
    par::Par
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
    fmsc::Blockscal
    weights::Weight
    niter::Vector{Int}
    par::Par
end

######---- Regression

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
    par::Union{ParPlsr, ParRrr}
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
    par::ParLwmlr
end

struct Lwmlr
    X::Matrix
    Y::Matrix
    par::ParLwmlr
end

struct Lwplsr
    X::Matrix
    Y::Matrix
    fm::Union{Nothing, Plsr}
    xscales::Vector
    par::ParLwplsr
end

struct LwplsrAvg
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
    par::ParSvmr
end

struct TreerDt
    fm::Union{DecisionTree.Root, DecisionTree.Ensemble}
    xscales::Vector
    featur::Vector{Int}
    par::Union{ParTreer, ParRfr}
end

## Multiblock

struct Mbplsr
    fm::Plsr
    T::Matrix
    R::Matrix
    C::Matrix
    fmsc::Blockscal
    ymeans::Vector
    yscales::Vector
    weights::Weight
    par::ParMbplsr
end

struct Mbplswest            # Used for mbplswest, mbwcov 
    T::Matrix
    P::Matrix
    R::Matrix
    W::Matrix
    C::Matrix
    Tbl::Vector{Matrix}
    Tb::Vector{Matrix}
    Pbl::Vector{Matrix}
    TT::Vector
    fmsc::Blockscal
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
    fmsc::Blockscal
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
    fmsc::Blockscal    
    yscales::Vector
    par::ParSoplsr
end

######---- Discrimination

struct Dmkern
    X::Matrix
    H::Matrix
    Hinv::Matrix
    detH::Float64
end

struct Dmnorm
    mu
    Uinv 
    detS
    cst
end

struct Dmnormlog
    mu
    Uinv 
    logdetS
    logcst
end

struct Mlrda
    fm::Mlr 
    lev::Vector
    ni::Vector{Int}
end

struct Lda
    fm::Vector{Dmnorm}
    W::Matrix  
    ct::Matrix
    priors::Vector
    ni::Vector{Int}
    lev::Vector
    weights::Weight
end

struct Qda
    fm::Vector{Dmnorm}
    Wi::AbstractVector  
    ct::Matrix
    priors::Vector
    ni::Vector{Int}
    lev::Vector
    weights::Weight
end

struct Kdeda
    fm::Vector{Dmkern}
    priors::AbstractVector
    lev::Vector
    ni::Vector{Int}
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
end

struct Plsrda
    fm::Union{Plsr, Splsr, Kplsr, Dkplsr}  
    lev::Vector
    ni::Vector{Int}
end

struct Plsprobda    
    fm::NamedTuple  
    lev::Vector
    ni::Vector{Int}
end

struct Rrda
    fm::Union{Rr, Krr}  
    lev::Vector
    ni::Vector{Int}
    par::Par
end

## Local
## (from below, fm not yet specified)

struct Knnda
    X::Matrix
    y::AbstractMatrix
    fm
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::Par
end

struct Lwmlrda
    X::Matrix
    y::AbstractMatrix
    lev::Vector
    ni::Vector{Int}
    par::Par
end

struct Lwplsrda
    X::Matrix
    y::AbstractMatrix
    fm
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::Par
end

struct Lwplslda
    X::Matrix
    y::AbstractMatrix
    fm
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::Par
end

struct Lwplsqda
    X::Matrix
    y::AbstractMatrix
    fm
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
    par::Par
end

## Svm, Trees

struct Svmda
    fm
    xscales::Vector
    lev::Vector
    ni::Vector{Int}
end

struct TreedaDt 
    fm
    xscales::Vector
    featur::Vector{Int}
    lev::Vector
    ni::Vector{Int}
    par::Par
end

## Multiblock

struct Mbplsrda
    fm::Mbplsr  
    lev::Vector
    ni::Vector{Int}
end

struct Mbplslda    # for Mbplslda and Mbplsqda 
    fm::NamedTuple  
    lev::Vector
    ni::Vector{Int}
end

## Occ

struct Occstah
    d::DataFrame
    res_stah::NamedTuple
    P::Matrix
    e_cdf::ECDF
    cutoff::Real
end

struct Occsd
    d::DataFrame 
    fm
    tscales::Vector
    e_cdf::ECDF
    cutoff::Real   
end

struct Occod
    d::DataFrame
    fm
    e_cdf::ECDF
    cutoff::Real   
end

struct Occsdod
    d::DataFrame
    fmsd
    fmod
end

######---- Data Processing 

## Preprocessing

struct Detrend
    par::Par
end

struct Fdif
    par::Par
end

struct Interpl
    par::Par
end

struct Mavg
    par::Par
end

struct Savgol
    par::Par
end

struct Snv
    par::Par
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
    par::Par
end

## Calibration transfer

struct CalDs
    fm
end

struct CalPds
    fm
    s
end
