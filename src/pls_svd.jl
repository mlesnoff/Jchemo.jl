# Tucker inter-battery
# Robust canonical analysis, Canonical covariances
# Tenenhaus, M., 1998. La régression PLS: théorie et pratique. 
# Editions Technip, Paris.
# Tishler, A., Lipovetsky, S., 2000. Modelling and forecasting with robust 
# canonical analysis: method and application. Computers & Operations Research 27, 
# 217–232. https://doi.org/10.1016/S0305-0548(99)00014-3
# Wegelin, J.A., 2000. A Survey of Partial Least Squares (PLS) Methods, 
# with Emphasis on the Two-Block Case. Technical report, University of Washington.

struct PlsSvd4
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

function pls_svd(X, Y, weights = ones(nro(X)); nlv,
        bscal = "none", scal = false)
    pls_svd!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv,
        bscal = bscal, scal = scal)
end

function pls_svd!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        bscal = "none", scal = false)
    n, p = size(X)
    q = nco(Y)
    nlv = min(nlv, n, p, q)
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)   
    xscales = ones(p)
    yscales = ones(q)
    if scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        cscale!(X, xmeans, xscales)
        cscale!(Y, ymeans, yscales)
    else
        center!(X, xmeans)
        center!(Y, ymeans)
    end
    bscal == "none" ? bscales = ones(2) : nothing
    if bscal == "frob"
        normx = fnorm(X, weights)
        normy = fnorm(Y, weights)
        X ./= normx
        Y ./= normy
        bscales = [normx; normy]
    end
    D = Diagonal(weights)
    XtY = X' * (D * Y)
    U, delta, V = svd(XtY)
    delta = delta[1:nlv]
    Wx = U[:, 1:nlv]
    Wy = V[:, 1:nlv]
    Tx = X * Wx
    Ty = Y * Wy
    TTx = colsum(D * Tx .* Tx)
    TTy = colsum(D * Ty .* Ty)
    PlsSvd4(Tx, Ty, Wx, Wy, TTx, TTy, delta, bscales, xmeans, xscales, 
        ymeans, yscales, weights)
end

function transform(object::PlsSvd4, X, Y; nlv = nothing)
    X = ensure_mat(X)
    Y = ensure_mat(Y)   
    a = nco(object.Tx)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    Tx = cscale(X, object.xmeans, object.xscales) * vcol(object.Wx, 1:nlv)
    Ty = cscale(Y, object.ymeans, object.yscales) * vcol(object.Wy, 1:nlv)
    (Tx = Tx, Ty)
end

function Base.summary(object::PlsSvd4, X::Union{Vector, Matrix, DataFrame},
        Y::Union{Vector, Matrix, DataFrame})
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, nlv = size(object.Tx)
    X = cscale(X, object.xmeans, object.xscales) / object.bscales[1]
    Y = cscale(Y, object.ymeans, object.yscales) / object.bscales[2]
    ttx = object.TTx
    tty = object.TTy
    ## X
    sstot = fnorm(X, object.weights)^2
    pvar = ttx / sstot
    cumpvar = cumsum(pvar)
    xvar = ttx / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    ## y
    sstot = fnorm(Y, object.weights)^2
    pvar = tty / sstot
    cumpvar = cumsum(pvar)
    xvar = tty / n    
    explvary = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)
    ## End
    (explvarx = explvarx, explvary)
end
