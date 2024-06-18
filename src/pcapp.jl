function pcapp(X; rob = true, nsim = 50, kwargs...)
    pcapp!(copy(ensure_mat(X)); rob, nsim, kwargs...)
end

function pcapp!(X::Matrix; rob = true, nsim = 50, kwargs...)
    par = recovkwargs(Par, kwargs) 
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    if rob 
        fcolmean = Jchemo.colmedspa
        fcolstd = colmad
        fobj = colstd
    else 
        fcolmean = colmean
        fcolstd = colstd
        fobj = colmad
    end
    xmeans = fcolmean(X) 
    xscales = ones(Q, p)
    if par.scal 
        xscales .= fcolstd(X)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    fsimpp = simppbin 
    for a = 1:nlv
        P = fsimpp(X; nsim)
        zT = X * P 
        zobj = colstd(zT)
        s = findall(zobj .== maximum(zobj))[1]
        sv = zobj[s]
        T[:, a] = vcol(T, a)
    end
    (T = T, P, sv, xmeans, xscales, kwargs, par) 
end

