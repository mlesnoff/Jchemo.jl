###### Dimension reduction

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
    kwargs::Base.Pairs
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
    kwargs::Base.Pairs
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
    bscales::Vector
    xmeans::Vector{Vector}
    xscales::Vector{Vector}
    weights::Weight
    niter::Vector{Int}
    kwargs::Base.Pairs
    par::Par
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
    lev::Vector
    ni::Vector{Int}
    kwargs::Base.Pairs
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
    kwargs::Base.Pairs
    par::Par
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
    bscales::Vector
    xmeans::Vector{Vector}
    xscales::Vector{Vector}
    weights::Weight
    niter::Vector{Int}
    kwargs::Base.Pairs
    par::Par
end

struct Pca 
    T::Matrix 
    P::Matrix
    sv::Vector
    xmeans::Vector
    xscales::Vector
    weights::Weight
    niter::Union{Vector{Int}, Nothing} # for PCA Nipals
    kwargs::Base.Pairs
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
    kwargs::Base.Pairs
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
    kwargs::Base.Pairs
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
    kwargs::Base.Pairs
    par::Par
end

struct Rp
    T::Matrix
    P::Matrix
    xmeans::Vector
    xscales::Vector
    kwargs::Base.Pairs
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
    kwargs::Base.Pairs
    par::Par
end

###### Regression

struct Cglsr
    B::Matrix
    g::Vector
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    F::Union{Matrix, Nothing}
    kwargs::Base.Pairs
    par::Par
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
    par::Par
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
    par::Par
end

struct Mbplsr
    fm
    T::Matrix
    R::Matrix
    C::Matrix
    bscales::Vector
    xmeans::Vector{Vector}
    xscales::Vector{Vector}
    ymeans::Vector
    yscales::Vector
    weights::Weight
    kwargs::Base.Pairs
    par::Par
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
    bscales::Vector
    xmeans::Vector{Vector}
    xscales::Vector{Vector}
    ymeans::Vector
    yscales::Vector
    weights::Weight
    lb::Union{Matrix, Nothing}
    niter::Union{Vector, Nothing}
    kwargs::Base.Pairs
    par::Par
end

struct Mlr
    B::Matrix   
    int::Matrix
    weights::Weight
    kwargs::Base.Pairs
    par::Par
end

struct MlrNoArg
    B::Matrix   
    int::Matrix
    weights::Weight
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
    kwargs::Base.Pairs
    par::Par
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
    kwargs::Base.Pairs
    par::Par
end

## Here since Plsr
struct Dkplsr
    X::Matrix
    fm::Plsr
    K::Matrix
    xscales::Vector
    yscales::Vector
    kwargs::Base.Pairs
    par::Par
end

struct PlsravgUnif
    fm::Plsr
    nlv::UnitRange
end

struct Plsravg
    fm::PlsravgUnif
    kwargs::Base.Pairs
    par::Par
end

struct Rosaplsr
    T::Matrix
    P::Matrix
    R::Matrix
    W::Matrix
    C::Matrix
    TT::Vector
    xmeans::Vector{Vector}
    xscales::Vector{Vector}
    ymeans::Vector
    yscales::Vector
    weights::Weight
    bl::Vector{Int}
    kwargs::Base.Pairs
    par::Par
end

struct Rr
    V::Matrix
    TtDY::Matrix
    sv::Vector
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    weights::Weight
    kwargs::Base.Pairs
    par::Par
end

struct Soplsr
    fm
    T::Matrix
    fit::Matrix
    b
    kwargs::Base.Pairs
    par::Par
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
    kwargs::Base.Pairs
    par::Par
end

struct Svmr
    fm::LIBSVM.SVM
    xscales::Vector
    kwargs::Base.Pairs
    par::Par
end

struct TreerDt
    fm::Union{DecisionTree.Root, DecisionTree.Ensemble}
    xscales::Vector
    featur::Vector{Int}
    mth::Bool 
    kwargs::Base.Pairs
    par::Par
end

## Local

struct Knnr
    X::Matrix
    Y::Matrix
    fm::Union{Nothing, Plsr}
    xscales::Vector
    kwargs::Base.Pairs
    par::Par
end

struct Lwmlr
    X::Matrix
    Y::Matrix
    kwargs::Base.Pairs
    par::Par
end

struct LwmlrS
    T::Matrix
    Y::Matrix
    fm::Union{Nothing, Pca, Plsr, Dkplsr}
    kwargs::Base.Pairs
    par::Par
end

struct Lwplsr
    X::Matrix
    Y::Matrix
    fm::Union{Nothing, Plsr}
    xscales::Vector
    kwargs::Base.Pairs
    par::Par
end

struct LwplsrAvg
    X::Matrix
    Y::Matrix
    fm::Union{Nothing, Plsr}
    xscales::Vector
    kwargs::Base.Pairs
    par::Par
end

struct LwplsrS
    T::Matrix
    Y::Matrix
    fm::Union{Nothing, Pca, Plsr, Dkplsr}
    kwargs::Base.Pairs
    par::Par
end

###### Discrimination

struct Dkplsrda
    fm  
    lev::Vector
    ni::Vector{Int}
end

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

struct Kdeda
    fm
    wprior::AbstractVector
    lev::Vector
    ni::Vector{Int}
end

struct Knnda
    X::Matrix
    y::AbstractMatrix
    fm
    nlvdis::Int
    metric::String
    h::Real
    k::Int
    tol::AbstractFloat
    lev::Vector
    ni::Vector{Int}
    scal::Bool
end

struct Lda
    fm
    W::Matrix  
    ct::Matrix
    wprior::Vector
    theta::Vector
    ni::Vector{Int}
    lev::Vector
    weights::Weight
end

struct Lwmlrda
    X::Matrix
    y::AbstractMatrix
    metric::String
    h::Real
    k::Int
    tol::AbstractFloat
    verbose::Bool
    kwargs::Base.Pairs
    par::Par
end

struct LwmlrdaS
    T::Matrix
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    tol::AbstractFloat
    verbose::Bool
    kwargs::Base.Pairs
    par::Par
end

struct Lwplslda
    X::Matrix
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    prior::String
    tol::AbstractFloat
    scal::Bool
    verbose::Bool
    lev::Vector
    ni::Vector{Int}
end

struct Lwplsqda
    X::Matrix
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    alpha::Real
    prior::String
    tol::AbstractFloat
    scal::Bool
    verbose::Bool
    lev::Vector
    ni::Vector{Int}
end

struct Lwplsrda
    X::Matrix
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    tol::AbstractFloat
    scal::Bool
    verbose::Bool
    lev::Vector
    ni::Vector{Int}
end

struct LwplsrdaS
    T::Matrix
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    tol::AbstractFloat
    scal::Bool
    verbose::Bool
    kwargs::Base.Pairs
    par::Par
end

struct Mlrda
    fm  
    lev::Vector
    ni::Vector{Int}
end

struct Plslda    # for plslda and plsqda 
    fm  
    lev::Vector
    ni::Vector{Int}
end

struct Plsrda
    fm  
    lev::Vector
    ni::Vector{Int}
end

struct Qda
    fm
    Wi::AbstractVector  
    ct::Matrix
    wprior::Vector
    theta::Vector
    ni::Vector{Int}
    lev::Vector
    weights::Weight
end

struct Rda
    fm
    Wi::AbstractVector  
    ct::Matrix
    wprior::Vector
    theta::Vector
    ni::Vector{Int}
    lev::Vector
    xscales::Vector
    weights::Weight
end

struct Rrda
    fm  
    lev::Vector
    ni::Vector{Int}
    kwargs::Base.Pairs
    par::Par
end

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
    mth::Bool 
    kwargs::Base.Pairs
    par::Par
end

## Occ

struct Occknndis
    d::DataFrame
    fm
    T::Matrix
    tscales::Vector
    k::Int
    e_cdf::ECDF
    cutoff::Real    
    kwargs::Base.Pairs
    par::Par
end

struct Occlknndis
    d::DataFrame
    fm
    T::Matrix
    tscales::Vector
    k::Int
    e_cdf::ECDF
    cutoff::Real    
    kwargs::Base.Pairs
    par::Par
end

struct Occod
    d
    fm
    e_cdf::ECDF
    cutoff::Real   
    nlv::Int
end

struct Occsd
    d
    fm
    Sinv::Matrix
    e_cdf::ECDF
    cutoff::Real   
    nlv::Int
end

struct Occsdod
    d::DataFrame
    fmsd
    fmod
end

struct Occstah
    d
    res_stah
    e_cdf::ECDF
    cutoff::Real
end

###### Preprocessing 

struct Detrend
    kwargs::Base.Pairs
    par::Par
end

struct Fdif
    kwargs::Base.Pairs
    par::Par
end

struct Interpl
    kwargs::Base.Pairs
    par::Par
end

struct Mavg
    kwargs::Base.Pairs
    par::Par
end

struct Savgol
    kwargs::Base.Pairs
    par::Par
end

struct Snv
    kwargs::Base.Pairs
    par::Par
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

