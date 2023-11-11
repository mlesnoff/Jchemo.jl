###### Dimension reduction

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
    weights::Vector{Q}
end

struct CcaWold{Q <: AbstractFloat}
    Tx::Matrix{Q}
    Ty::Matrix{Q}
    Px::Matrix{Q}
    Py::Matrix{Q}
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
    weights::Vector{Q}
    niter::Vector{Int}
end

struct Comdim{Q <: AbstractFloat}
    T::Array{Q} 
    U::Array{Q}
    W::Array{Q}
    Tbl::Vector{Array{Q}}
    Tb::Vector{Array{Q}}
    Wbl::Vector{Array{Q}}
    lb::Array{Q}
    mu::Vector{Q}
    bscales::Vector{Q}
    xmeans::Vector{Vector{Q}}
    xscales::Vector{Vector{Q}}
    weights::Vector{Q}
    niter::Vector{Int}
end

struct Fda{Q <: AbstractFloat}
    T::Array{Q}
    P::Array{Q}
    Tcenters::Array{Q}
    eig::Vector{Q}
    sstot::Number
    W::Matrix{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    lev::AbstractVector
    ni::AbstractVector
end

struct Kpca{Q <: AbstractFloat}
    X::Array{Q}
    Kt::Array{Q}
    T::Array{Q}
    P::Array{Q}
    sv::Vector{Q}  
    eig::Vector{Q}    
    D::Array{Q} 
    DKt::Array{Q}
    vtot::Array{Q}
    xscales::Vector{Q} 
    weights::Vector{Q}
    kern
    dots
end

struct Mbpca{Q <: AbstractFloat}
    T::Array{Q} 
    U::Array{Q}
    W::Array{Q}
    Tbl::Vector{Array{Q}}
    Tb::Vector{Array{Q}}
    Wbl::Vector{Array{Q}}
    lb::Array{Q}
    mu::Vector{Q}
    bscales::Vector{Q}
    xmeans::Vector{Vector{Q}}
    xscales::Vector{Vector{Q}}
    weights::Vector{Q}
    niter::Vector{Int}
end

struct MbplsWest{Q <: AbstractFloat}            # Used for mbplswest, mbwcov 
    T::Matrix{Q}
    P::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    Tbl::Vector{Array{Q}}
    Tb::Vector{Array{Q}}
    Pbl::Vector{Array{Q}}
    TT::Vector{Q}
    bscales::Vector{Q}
    xmeans::Vector{Vector{Q}}
    xscales::Vector{Vector{Q}}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::Vector{Q}
    lb::Union{Array{Q}, Nothing}
    niter::Union{Array{Q}, Nothing}
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

struct PlsCan{Q <: AbstractFloat}
    Tx::Matrix{Q}
    Ty::Matrix{Q}
    Px::Matrix{Q}
    Py::Matrix{Q}
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
    weights::Vector{Q}
end

struct PlsTuck{Q <: AbstractFloat}
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
    weights::Vector{Q}
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
    weights::Vector{Q}
end

struct Rp{Q <: AbstractFloat}
    T::Matrix{Q}
    P
    xmeans
    xscales
end

struct Spca{Q <: AbstractFloat}
    T::Array{Q} 
    P::Array{Q}
    sv::Vector{Q}
    beta::Array{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    weights::Vector{Q}
    niter::Union{Vector{Int}, Nothing}
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
end

###### Regression

struct Cglsr{Q <: AbstractFloat}
    B::Matrix{Q}
    g::Vector{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    F::Union{Array{Q}, Nothing}
end

struct Covselr{Q <: AbstractFloat}
    T::Matrix{Q}
    P::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    TT::Vector{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::Vector{Q}
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
end

struct Knnr{Q <: AbstractFloat}
    X::Array{Q}
    Y::Array{Q}
    fm
    nlvdis::Int
    metric::String
    h::Real
    k::Int
    tol::Real
    scal::Bool
end

struct Kplsr{Q <: AbstractFloat}
    X::Matrix{Q}
    Kt::Adjoint{Q, Matrix{Q}}
    T::Matrix{Q}
    C::Matrix{Q}
    U::Matrix{Q}
    R::Matrix{Q}
    D::Diagonal{Q, Vector{Q}} 
    DKt::Matrix{Q}
    vtot::Matrix{Q}   
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::Vector{Q}
    iter::Vector{Int}
    par::Par
end

struct Krr{Q <: AbstractFloat}
    X::Matrix{Q}
    K::Matrix{Q}
    U::Matrix{Q}
    UtDY::Matrix{Q}
    sv::Vector{Q}
    D::Diagonal{Q, Vector{Q}}
    sqrtD::Diagonal{Q, Vector{Q}}
    DKt::Matrix{Q}
    vtot::Matrix{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    weights::Vector{Q}
    par::Par
end

struct Lwmlr{Q <: AbstractFloat}
    X::Array{Q}
    Y::Array{Q}
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

struct LwmlrS{Q <: AbstractFloat}
    T::Array{Q}
    Y::Array{Q}
    fm
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

struct Lwplsr{Q <: AbstractFloat}
    X::Array{Q}
    Y::Array{Q}
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    tol::Real
    scal::Bool
    verbose::Bool
end

struct LwplsrAvg{Q <: AbstractFloat}
    X::Array{Q}
    Y::Array{Q}
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

struct LwplsrS{Q <: AbstractFloat}
    T::Array{Q}
    Y::Array{Q}
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    tol::Real
    scal::Bool
    verbose::Bool
end

struct Mbplsr{Q <: AbstractFloat}
    fm
    T::Matrix{Q}
    R::Matrix{Q}
    C::Matrix{Q}
    bscales::Vector{Q}
    xmeans::Vector{Vector{Q}}
    xscales::Vector{Vector{Q}}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::Vector{Q}
end

struct Mlr{Q <: AbstractFloat}
    B::Matrix{Q}   
    int::Matrix{Q}
    weights::Vector{Q}
end

struct Pcr{Q <: AbstractFloat}
    fm_pca
    T::Matrix{Q}
    R::Matrix{Q}
    C::Matrix{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::Vector{Q}
end

struct Plsr{Q <: AbstractFloat}
    T::Matrix{Q}
    P::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    TT::Vector{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::Vector{Q}
    niter::Union{Vector{Int}, Nothing}
end

struct Plsravg
    fm
end

struct PlsravgCri{Q <: AbstractFloat}
    fm::Plsr
    nlv
    w::Vector{Q}
end

struct PlsravgShenk
    fm::Plsr
    nlv
end

struct PlsravgUnif
    fm::Plsr
    nlv
end

struct Plsrstack{Q <: AbstractFloat}
    fm::Plsr
    nlv
    w::Vector{Q}
    Xstack  # = View
    ystack::Array{Q}
    weightsstack::Array{Q}
end

struct Rosaplsr{Q <: AbstractFloat}
    T::Matrix{Q}
    P::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    TT::Vector{Q}
    xmeans::Vector{Vector{Q}}
    xscales::Vector{Vector{Q}}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::Vector{Q}
    bl::Vector
end

struct Rr{Q <: AbstractFloat}
    V::Array{Q}
    TtDY::Array{Q}
    sv::Vector{Q}
    lb::Float64
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    weights::Vector{Q}
end

struct Soplsr{Q <: AbstractFloat}
    fm
    T::Matrix{Q}
    fit::Matrix{Q}
    b
end

struct Splsr{Q <: AbstractFloat}
    T::Matrix{Q}
    P::Matrix{Q}
    R::Matrix{Q}
    W::Matrix{Q}
    C::Matrix{Q}
    TT::Vector{Q}
    xmeans::Vector{Q}
    xscales::Vector{Q}
    ymeans::Vector{Q}
    yscales::Vector{Q}
    weights::Vector{Q}
    niter::Union{Array{Q}, Nothing}
    sellv::Vector{Vector{Int}}
    sel::Vector{Int}
end

struct Svmr{Q <: AbstractFloat}
    fm
    xscales::Vector{Q}
end

struct TreerDt{Q <: AbstractFloat}
    fm
    xscales::Vector{Q}
    featur::Vector{Int}
    mth::Bool 
end

###### Discrimination

struct Dkplsrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

struct Dmkern{Q <: AbstractFloat}
    X::Array{Q}
    H::Array{Q}
    Hinv::Array{Q}
    detH::Float64
end

struct Dmnorm{Q <: AbstractFloat}
    mu
    Uinv 
    detS
    cst
end

struct Dmnormlog{Q <: AbstractFloat}
    mu
    Uinv 
    logdetS
    logcst
end

struct Kernda{Q <: AbstractFloat}
    fm
    wprior::AbstractVector
    lev::AbstractVector
    ni::AbstractVector
end

struct Knnda{Q <: AbstractFloat}
    X::Array{Q}
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

struct Lda{Q <: AbstractFloat}
    fm
    W::Array{Q}  
    ct::Array{Q}
    wprior::Vector{Q}
    theta::Vector{Q}
    ni::Vector{Int}
    lev::AbstractVector
    weights::Vector{Q}
end

struct Lwmlrda{Q <: AbstractFloat}
    X::Array{Q}
    y::AbstractMatrix
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

struct LwmlrdaS{Q <: AbstractFloat}
    T::Array{Q}
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

struct Lwplslda{Q <: AbstractFloat}
    X::Array{Q}
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

struct LwplsldaAvg{Q <: AbstractFloat}
    X::Array{Q}
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

struct Lwplsqda{Q <: AbstractFloat}
    X::Array{Q}
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

struct LwplsqdaAvg{Q <: AbstractFloat}
    X::Array{Q}
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

struct Lwplsrda{Q <: AbstractFloat}
    X::Array{Q}
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

struct LwplsrdaAvg{Q <: AbstractFloat}
    X::Array{Q}
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

struct LwplsrdaS{Q <: AbstractFloat}
    T::Array{Q}
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

struct Mlrda{Q <: AbstractFloat}
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

struct Nscda{Q <: AbstractFloat}
    fms
    poolstd_s0::Vector{Q}
    wprior::Vector{Q}
    ni::Vector{Int}
    lev::AbstractVector
    xscales::Vector{Q}
    weights::Vector{Q}
end

struct Occknndis{Q <: AbstractFloat}
    d::DataFrame
    fm
    T::Array{Q}
    tscales::Vector{Q}
    k::Int
    e_cdf::ECDF
    cutoff::Real    
end

struct Occlknndis{Q <: AbstractFloat}
    d::DataFrame
    fm
    T::Array{Q}
    tscales::Vector{Q}
    k::Int
    e_cdf::ECDF
    cutoff::Real    
end

struct Occod{Q <: AbstractFloat}
    d
    fm
    e_cdf::ECDF
    cutoff::Real   
    nlv::Int
end

struct Occsd{Q <: AbstractFloat}
    d
    fm
    Sinv::Matrix{Q}
    e_cdf::ECDF
    cutoff::Real   
    nlv::Int
end

struct Occsdod{Q <: AbstractFloat}
    d::DataFrame
    fm_sd
    fm_od
end

struct Occstah{Q <: AbstractFloat}
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

struct Plsrdaavg{Q <: AbstractFloat}  # for plsrdaavg, plsldaavg and plsqdaavg 
    fm
    nlv
    w_mod
    lev::AbstractVector
    ni::AbstractVector
end

struct Qda{Q <: AbstractFloat}
    fm
    Wi::AbstractVector  
    ct::Array{Q}
    wprior::Vector{Q}
    theta::Vector{Q}
    ni::Vector{Int}
    lev::AbstractVector
    weights::Vector{Q}
end

struct Rda{Q <: AbstractFloat}
    fm
    Wi::AbstractVector  
    ct::Array{Q}
    wprior::Vector{Q}
    theta::Vector{Q}
    ni::Vector{Int}
    lev::AbstractVector
    xscales::Vector{Q}
    weights::Vector{Q}
end

struct Rrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

struct Svmda{Q <: AbstractFloat}
    fm
    xscales::Vector{Q}
    lev::AbstractVector
    ni::AbstractVector
end

struct TreedaDt{Q <: AbstractFloat} 
    fm
    xscales::Vector{Q}
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

struct Dkplsr{Q <: AbstractFloat}
    X::Matrix{Q}
    fm::Plsr
    K::Matrix{Q}
    xscales::Vector{Q}
    yscales::Vector{Q}
    par::Par
end

