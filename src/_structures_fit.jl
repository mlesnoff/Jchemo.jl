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
    weights::Vector
end

struct CcaWold
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
    weights::Vector
    niter::Vector{Int}
end

struct Comdim
    T::Array 
    U::Array
    W::Array
    Tbl::Vector{Array}
    Tb::Vector{Array}
    Wbl::Vector{Array}
    lb::Array
    mu::Vector
    bscales::Vector
    xmeans::Vector{Vector}
    xscales::Vector{Vector}
    weights::Vector
    niter::Vector{Int}
end

struct Fda
    T::Array
    P::Array
    Tcenters::Array
    eig::Vector
    sstot::Number
    W::Matrix
    xmeans::Vector
    xscales::Vector
    lev::AbstractVector
    ni::AbstractVector
end

struct Kpca
    X::Array
    Kt::Array
    T::Array
    P::Array
    sv::Vector  
    eig::Vector    
    D::Array 
    DKt::Array
    vtot::Array
    xscales::Vector 
    weights::Vector
    kern
    dots
end

struct Mbpca
    T::Array 
    U::Array
    W::Array
    Tbl::Vector{Array}
    Tb::Vector{Array}
    Wbl::Vector{Array}
    lb::Array
    mu::Vector
    bscales::Vector
    xmeans::Vector{Vector}
    xscales::Vector{Vector}
    weights::Vector
    niter::Vector{Int}
end

struct MbplsWest            # Used for mbplswest, mbwcov 
    T::Matrix
    P::Matrix
    R::Matrix
    W::Matrix
    C::Matrix
    Tbl::Vector{Array}
    Tb::Vector{Array}
    Pbl::Vector{Array}
    TT::Vector
    bscales::Vector
    xmeans::Vector{Vector}
    xscales::Vector{Vector}
    ymeans::Vector
    yscales::Vector
    weights::Vector
    lb::Union{Array, Nothing}
    niter::Union{Array, Nothing}
end

struct Pca 
    T::Matrix 
    P::Matrix
    sv::Vector
    xmeans::Vector
    xscales::Vector
    weights::Vector
    ## For PCA Nipals
    niter::Union{Vector{Int}, Nothing}
end

struct PlsCan
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
    weights::Vector
end

struct PlsTuck
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
    weights::Vector
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
    weights::Vector
end

struct Rp
    T::Matrix
    P
    xmeans
    xscales
end

struct Spca
    T::Array 
    P::Array
    sv::Vector
    beta::Array
    xmeans::Vector
    xscales::Vector
    weights::Vector
    niter::Union{Vector{Int}, Nothing}
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
end

###### Regression

struct Cglsr
    B::Matrix
    g::Vector
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    F::Union{Array, Nothing}
end

struct Covselr
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
    weights::Vector
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
end

struct Knnr
    X::Array
    Y::Array
    fm
    nlvdis::Int
    metric::String
    h::Real
    k::Int
    tol::Real
    scal::Bool
end

struct Kplsr
    X::Matrix
    Kt::Adjoint{Matrix}
    T::Matrix
    C::Matrix
    U::Matrix
    R::Matrix
    D::Diagonal{Vector} 
    DKt::Matrix
    vtot::Matrix   
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Vector
    iter::Vector{Int}
    par::Par
end

struct Krr
    X::Matrix
    K::Matrix
    U::Matrix
    UtDY::Matrix
    sv::Vector
    D::Diagonal{Vector}
    sqrtD::Diagonal{Vector}
    DKt::Matrix
    vtot::Matrix
    xscales::Vector
    ymeans::Vector
    weights::Vector
    par::Par
end

struct Lwmlr
    X::Array
    Y::Array
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

struct LwmlrS
    T::Array
    Y::Array
    fm
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

struct Lwplsr
    X::Array
    Y::Array
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    tol::Real
    scal::Bool
    verbose::Bool
end

struct LwplsrAvg
    X::Array
    Y::Array
    fm
    metric::String
    h::Real
    k::Int
    nlv::String
    typf::String
    typw::String
    alpha::Real
    K::Real
    rep::Real
    tol::Real
    scal::Bool
    verbose::Bool
end

struct LwplsrS
    T::Array
    Y::Array
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    tol::Real
    scal::Bool
    verbose::Bool
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
    weights::Vector
end

struct Mlr
    B::Matrix   
    int::Matrix
    weights::Vector
end

struct Pcr
    fm_pca
    T::Matrix
    R::Matrix
    C::Matrix
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    yscales::Vector
    weights::Vector
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
    weights::Vector
    niter::Union{Vector{Int}, Nothing}
end

struct Plsravg
    fm
end

struct PlsravgCri
    fm::Plsr
    nlv
    w::Vector
end

struct PlsravgShenk
    fm::Plsr
    nlv
end

struct PlsravgUnif
    fm::Plsr
    nlv
end

struct Plsrstack
    fm::Plsr
    nlv
    w::Vector
    Xstack  # = View
    ystack::Array
    weightsstack::Array
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
    weights::Vector
    bl::Vector
end

struct Rr
    V::Array
    TtDY::Array
    sv::Vector
    lb::Float64
    xmeans::Vector
    xscales::Vector
    ymeans::Vector
    weights::Vector
end

struct Soplsr
    fm
    T::Matrix
    fit::Matrix
    b
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
    weights::Vector
    niter::Union{Array, Nothing}
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
end

struct Svmr
    fm
    xscales::Vector
end

struct TreerDt
    fm
    xscales::Vector
    featur::Vector{Int}
    mth::Bool 
end

###### Discrimination

struct Dkplsrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

struct Dmkern
    X::Array
    H::Array
    Hinv::Array
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

struct Kernda
    fm
    wprior::AbstractVector
    lev::AbstractVector
    ni::AbstractVector
end

struct Knnda
    X::Array
    y::AbstractMatrix
    fm
    nlvdis::Int
    metric::String
    h::Real
    k::Int
    tol::Real
    lev::AbstractVector
    ni::AbstractVector
    scal::Bool
end

struct Lda
    fm
    W::Array  
    ct::Array
    wprior::Vector
    theta::Vector
    ni::Vector{Int}
    lev::AbstractVector
    weights::Vector
end

struct Lwmlrda
    X::Array
    y::AbstractMatrix
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

struct LwmlrdaS
    T::Array
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

struct Lwplslda
    X::Array
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    prior::String
    tol::Real
    scal::Bool
    verbose::Bool
    lev::AbstractVector
    ni::AbstractVector
end

struct LwplsldaAvg
    X::Array
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::String
    tol::Real
    scal::Bool
    verbose::Bool
    lev::AbstractVector
    ni::AbstractVector
end

struct Lwplsqda
    X::Array
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    alpha::Real
    prior::String
    tol::Real
    scal::Bool
    verbose::Bool
    lev::AbstractVector
    ni::AbstractVector
end

struct LwplsqdaAvg
    X::Array
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::String
    alpha::Real
    tol::Real
    scal::Bool
    verbose::Bool
    lev::AbstractVector
    ni::AbstractVector
end

struct Lwplsrda
    X::Array
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    tol::Real
    scal::Bool
    verbose::Bool
    lev::AbstractVector
    ni::AbstractVector
end

struct LwplsrdaAvg
    X::Array
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::String
    tol::Real
    scal::Bool
    verbose::Bool
    lev::AbstractVector
    ni::AbstractVector
end

struct LwplsrdaS
    T::Array
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    tol::Real
    scal::Bool
    verbose::Bool
end

struct Mlrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

struct Nscda
    fms
    poolstd_s0::Vector
    wprior::Vector
    ni::Vector{Int}
    lev::AbstractVector
    xscales::Vector
    weights::Vector
end

struct Occknndis
    d::DataFrame
    fm
    T::Array
    tscales::Vector
    k::Int
    e_cdf::ECDF
    cutoff::Real    
end

struct Occlknndis
    d::DataFrame
    fm
    T::Array
    tscales::Vector
    k::Int
    e_cdf::ECDF
    cutoff::Real    
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
    fm_sd
    fm_od
end

struct Occstah
    d
    res_stah
    e_cdf::ECDF
    cutoff::Real
end

struct Plslda    # for plslda and plsqda 
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

struct Plsrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

struct Plsrdaavg  # for plsrdaavg, plsldaavg and plsqdaavg 
    fm
    nlv
    w_mod
    lev::AbstractVector
    ni::AbstractVector
end

struct Qda
    fm
    Wi::AbstractVector  
    ct::Array
    wprior::Vector
    theta::Vector
    ni::Vector{Int}
    lev::AbstractVector
    weights::Vector
end

struct Rda
    fm
    Wi::AbstractVector  
    ct::Array
    wprior::Vector
    theta::Vector
    ni::Vector{Int}
    lev::AbstractVector
    xscales::Vector
    weights::Vector
end

struct Rrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

struct Svmda
    fm
    xscales::Vector
    lev::AbstractVector
    ni::AbstractVector
end

struct TreedaDt 
    fm
    xscales::Vector
    featur::Vector{Int}
    lev::AbstractVector
    ni::AbstractVector
    mth::Bool 
end

###### Related

struct CplsrAvg
    fm
    fm_da::Plslda
    lev
    ni
end

struct Dkplsr
    X::Matrix
    fm::Plsr
    K::Matrix
    xscales::Vector
    yscales::Vector
    par::Par
end

