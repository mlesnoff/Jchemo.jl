function pcapp(X; rob = true, nsim = 2000, kwargs...)
    pcapp!(copy(ensure_mat(X)); rob, nsim, kwargs...)
end

function pcapp!(X::Matrix; rob = true, nsim = 2000, kwargs...)
    par = recovkwargs(Par, kwargs) 
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    if rob 
        fcolmean = Jchemo.colmedspa
        fcolstd = colmad
        fobj = colmad
    else 
        fcolmean = colmean
        fcolstd = colstd
        fobj = colstd
    end
    xmeans = fcolmean(X) 
    xscales = ones(Q, p)
    if par.scal 
        xscales .= fcolstd(X)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    T = similar(X, n, nlv)
    P = similar(X, p, nlv)
    t = similar(X, n)
    zp = similar(X, p)
    sv = similar(X, nlv)
    fsimpp = simpphub
    #fsimpp = simppsph  # ~ same for large nsim (~ >= 2000)
    for a = 1:nlv
        zP = fsimpp(X; nsim)  # for simpphub: the nb. columns of P is variable
        #zP = zP[:, (n + 1):end] 
        zT = X * zP 
        zobj = fobj(zT)
        zobj[isnan.(zobj)] .= 0
        s = findall(zobj .== maximum(zobj))[1]
        sv[a] = zobj[s]
        t .= vcol(zT, s)
        zp .= vcol(zP, s)
        T[:, a] = t
        P[:, a] = zp
        X .-= t * zp'
    end
    s = sortperm(sv; rev = true)
    T .= T[:, s]
    P .= P[:, s]
    sv .= sv[s]
    weights = mweight(ones(n))
    Pca(T, P, sv, xmeans, xscales, weights, nothing, kwargs, par)
end

