###### Dimension reduction

struct Cca
    Tx::Matrix{Float64}
    Ty::Matrix{Float64}
    Wx::Matrix{Float64}
    Wy::Matrix{Float64}
    d::Vector{Float64}    
    bscales::Vector{Float64}    
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
end

struct CcaWold
    Tx::Matrix{Float64}
    Ty::Matrix{Float64}
    Px::Matrix{Float64}
    Py::Matrix{Float64}
    Rx::Matrix{Float64}
    Ry::Matrix{Float64}    
    Wx::Matrix{Float64}
    Wy::Matrix{Float64}
    TTx::Vector{Float64}
    TTy::Vector{Float64}  
    bscales::Vector{Float64}    
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
    niter::Vector{Float64}
end

struct Comdim
    T::Array{Float64} 
    U::Array{Float64}
    W::Array{Float64}
    Tbl::Vector{Array{Float64}}
    Tb::Vector{Array{Float64}}
    Wbl::Vector{Array{Float64}}
    lb::Array{Float64}
    mu::Vector{Float64}
    bscales::Vector{Float64}
    xmeans::Vector{Vector{Float64}}
    xscales::Vector{Vector{Float64}}
    weights::Vector{Float64}
    niter::Vector{Float64}
end

struct Fda
    T::Array{Float64}
    P::Array{Float64}
    Tcenters::Array{Float64}
    eig::Vector{Float64}
    sstot::Number
    W::Matrix{Float64}
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    lev::AbstractVector
    ni::AbstractVector
end

struct Kpca
    X::Array{Float64}
    Kt::Array{Float64}
    T::Array{Float64}
    P::Array{Float64}
    sv::Vector{Float64}  
    eig::Vector{Float64}    
    D::Array{Float64} 
    DKt::Array{Float64}
    vtot::Array{Float64}
    xscales::Vector{Float64} 
    weights::Vector{Float64}
    kern
    dots
end

struct Mbpca
    T::Array{Float64} 
    U::Array{Float64}
    W::Array{Float64}
    Tbl::Vector{Array{Float64}}
    Tb::Vector{Array{Float64}}
    Wbl::Vector{Array{Float64}}
    lb::Array{Float64}
    mu::Vector{Float64}
    bscales::Vector{Float64}
    xmeans::Vector{Vector{Float64}}
    xscales::Vector{Vector{Float64}}
    weights::Vector{Float64}
    niter::Vector{Float64}
end

struct MbplsWest            # Used for mbplswest, mbwcov 
    T::Matrix{Float64}
    P::Matrix{Float64}
    R::Matrix{Float64}
    W::Matrix{Float64}
    C::Matrix{Float64}
    Tbl::Vector{Array{Float64}}
    Tb::Vector{Array{Float64}}
    Pbl::Vector{Array{Float64}}
    TT::Vector{Float64}
    bscales::Vector{Float64}
    xmeans::Vector{Vector{Float64}}
    xscales::Vector{Vector{Float64}}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
    lb::Union{Array{Float64}, Nothing}
    niter::Union{Array{Float64}, Nothing}
end

struct Pca
    T::Array{Float64} 
    P::Array{Float64}
    sv::Vector{Float64}
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    weights::Vector{Float64}
    ## For PCA Nipals
    niter::Union{Vector{Int64}, Nothing}
end

struct PlsCan
    Tx::Matrix{Float64}
    Ty::Matrix{Float64}
    Px::Matrix{Float64}
    Py::Matrix{Float64}
    Rx::Matrix{Float64}
    Ry::Matrix{Float64}    
    Wx::Matrix{Float64}
    Wy::Matrix{Float64}
    TTx::Vector{Float64}
    TTy::Vector{Float64}
    delta::Vector{Float64}    
    bscales::Vector{Float64}    
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
end

struct PlsTuck
    Tx::Matrix{Float64}
    Ty::Matrix{Float64}
    Wx::Matrix{Float64}
    Wy::Matrix{Float64}
    TTx::Vector{Float64}
    TTy::Vector{Float64}
    delta::Vector{Float64}
    bscales::Vector{Float64}    
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
end

struct Rasvd
    Tx::Matrix{Float64}
    Ty::Matrix{Float64}
    Bx::Matrix{Float64}
    Wy::Matrix{Float64}
    lambda::Vector{Float64}    
    bscales::Vector{Float64}    
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
end

struct Rp
    T::Matrix{Float64}
    P
    xmeans
    xscales
end

struct Spca
    T::Array{Float64} 
    P::Array{Float64}
    sv::Vector{Float64}
    beta::Array{Float64}
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    weights::Vector{Float64}
    niter::Union{Vector{Int64}, Nothing}
    sellv::Vector{Vector{Int64}}
    sel::Vector{Int64}
end

###### Regression

struct Cglsr
    B::Matrix{Float64}
    g::Vector{Float64}
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    F::Union{Array{Float64}, Nothing}
end

struct Covselr
    T::Matrix{Float64}
    P::Matrix{Float64}
    R::Matrix{Float64}
    W::Matrix{Float64}
    C::Matrix{Float64}
    TT::Vector{Float64}
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
    sellv::Vector{Vector{Int64}}
    sel::Vector{Int64}
end

struct Dkplsr
    X::Array{Float64}
    fm
    K::Array{Float64}
    kern
    xscales::Vector{Float64}
    yscales::Vector{Float64}
    dots
end

struct Knnr
    X::Array{Float64}
    Y::Array{Float64}
    fm
    nlvdis::Int
    metric::String
    h::Real
    k::Int
    tol::Real
    scal::Bool
end

struct Kplsr
    X::Array{Float64}
    Kt::Array{Float64}
    T::Array{Float64}
    C::Array{Float64}
    U::Array{Float64}
    R::Array{Float64}
    D::Array{Float64} 
    DKt::Array{Float64}
    vtot::Array{Float64}   
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
    kern
    dots
    iter::Vector{Int}
end

struct Krr
    X::Array{Float64}
    K::Array{Float64}
    U::Array{Float64}
    UtDY::Array{Float64}
    sv::Vector{Float64}
    D::Array{Float64}
    sqrtD::Array{Float64}
    DKt::Array{Float64}
    vtot::Array{Float64}
    lb::Float64
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    weights::Vector{Float64}
    kern
    dots
end

struct Lwmlr
    X::Array{Float64}
    Y::Array{Float64}
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

struct LwmlrS
    T::Array{Float64}
    Y::Array{Float64}
    fm
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

struct Lwplsr
    X::Array{Float64}
    Y::Array{Float64}
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
    X::Array{Float64}
    Y::Array{Float64}
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
    T::Array{Float64}
    Y::Array{Float64}
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
    T::Matrix{Float64}
    R::Matrix{Float64}
    C::Matrix{Float64}
    bscales::Vector{Float64}
    xmeans::Vector{Vector{Float64}}
    xscales::Vector{Vector{Float64}}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
end

struct Mlr
    B::Matrix{Float64}   
    int::Matrix{Float64}
    weights::Vector{Float64}
end

struct Pcr
    fm_pca
    T::Matrix{Float64}
    R::Matrix{Float64}
    C::Matrix{Float64}
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
end

struct Plsr
    T::Matrix{Float64}
    P::Matrix{Float64}
    R::Matrix{Float64}
    W::Matrix{Float64}
    C::Matrix{Float64}
    TT::Vector{Float64}
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
    niter::Union{Array{Float64}, Nothing}
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
    T::Matrix{Float64}
    P::Matrix{Float64}
    R::Matrix{Float64}
    W::Matrix{Float64}
    C::Matrix{Float64}
    TT::Vector{Float64}
    xmeans::Vector{Vector{Float64}}
    xscales::Vector{Vector{Float64}}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
    bl::Vector
end

struct Rr
    V::Array{Float64}
    TtDY::Array{Float64}
    sv::Vector{Float64}
    lb::Float64
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    weights::Vector{Float64}
end

struct Soplsr
    fm
    T::Matrix{Float64}
    fit::Matrix{Float64}
    b
end

struct Splsr
    T::Matrix{Float64}
    P::Matrix{Float64}
    R::Matrix{Float64}
    W::Matrix{Float64}
    C::Matrix{Float64}
    TT::Vector{Float64}
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
    niter::Union{Array{Float64}, Nothing}
    sellv::Vector{Vector{Int64}}
    sel::Vector{Int64}
end

struct Svmr
    fm
    xscales::Vector{Float64}
end

struct TreerDt
    fm
    xscales::Vector{Float64}
    featur::Vector{Int64}
    mth::Bool 
end

###### Discrimination

struct Dkplsrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

struct Dmkern
    X::Array{Float64}
    H::Array{Float64}
    Hinv::Array{Float64}
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
    X::Array{Float64}
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
    W::Array{Float64}  
    ct::Array{Float64}
    wprior::Vector{Float64}
    theta::Vector{Float64}
    ni::Vector{Int64}
    lev::AbstractVector
    weights::Vector{Float64}
end

struct Lwmlrda
    X::Array{Float64}
    y::AbstractMatrix
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

struct LwmlrdaS
    T::Array{Float64}
    y::AbstractMatrix
    fm
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

struct Lwplslda
    X::Array{Float64}
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
    X::Array{Float64}
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
    X::Array{Float64}
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
    X::Array{Float64}
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
    X::Array{Float64}
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
    X::Array{Float64}
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
    T::Array{Float64}
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
    poolstd_s0::Vector{Float64}
    wprior::Vector{Float64}
    ni::Vector{Int64}
    lev::AbstractVector
    xscales::Vector{Float64}
    weights::Vector{Float64}
end

struct Occknndis
    d::DataFrame
    fm
    T::Array{Float64}
    tscales::Vector{Float64}
    k::Int
    e_cdf::ECDF
    cutoff::Real    
end

struct Occlknndis
    d::DataFrame
    fm
    T::Array{Float64}
    tscales::Vector{Float64}
    k::Int
    e_cdf::ECDF
    cutoff::Real    
end

struct Occod
    d
    fm
    e_cdf::ECDF
    cutoff::Real   
    nlv::Int64
end

struct Occsd
    d
    fm
    Sinv::Matrix{Float64}
    e_cdf::ECDF
    cutoff::Real   
    nlv::Int64
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

struct Plsdaavg  # for plsrdaavg, plsldaavg and plsqdaavg 
    fm
    nlv
    w_mod
    lev::AbstractVector
    ni::AbstractVector
end

struct Qda
    fm
    Wi::AbstractVector  
    ct::Array{Float64}
    wprior::Vector{Float64}
    theta::Vector{Float64}
    ni::Vector{Int64}
    lev::AbstractVector
    weights::Vector{Float64}
end

struct Rda
    fm
    Wi::AbstractVector  
    ct::Array{Float64}
    wprior::Vector{Float64}
    theta::Vector{Float64}
    ni::Vector{Int64}
    lev::AbstractVector
    xscales::Vector{Float64}
    weights::Vector{Float64}
end

struct Rrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

struct Svmda
    fm
    xscales::Vector{Float64}
    lev::AbstractVector
    ni::AbstractVector
end

struct TreedaDt 
    fm
    xscales::Vector{Float64}
    featur::Vector{Int64}
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


