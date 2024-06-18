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
    sv = similar(X, nsim)
    fsimpp = simppbin 
    #fsimpp = simpphub
    for a = 1:nlv
        zP = fsimpp(X; nsim)  # variable nb. columns for simpphub
        zT = X * zP 
        zobj = colstd(zT)
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
    (T = T, P, sv, xmeans, xscales, kwargs, par) 
end

