function pcapp(X, weights::Weight; rob = true, kwargs...)
    pcapp!(copy(ensure_mat(X)), weights; rob, kwargs...)
end

function pcapp!(X::Matrix, weights::Weight; rob = true, kwargs...)
    par = recovkwargs(Par, kwargs) 
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    if rob 
        fcolmean = colmean
        fcolsdt = colstd
    else 
        fcolmean = colmean
        fcolstd = colmad
    end
    xmeans = fcolmean(X, weights) 
    xscales = ones(Q, p)
    if par.scal 
        xscales .= fcolstd(X, weights)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end


    sqrtw = sqrt.(weights.w)
    X .= Diagonal(sqrtw) * X
    res = eigen!(Symmetric(X' * X); sortby = x -> -abs(x)) 
    P = res.vectors[:, 1:nlv]
    eig = res.values[1:min(n, p)]
    eig[eig .< 0] .= 0
    sv = sqrt.(eig)
    T = Diagonal(1 ./ sqrtw) * X * P
    Pca(T, P, sv, xmeans, xscales, weights, nothing, kwargs, par) 
end

