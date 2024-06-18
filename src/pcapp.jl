function pcapp(X; nsim = 2000, kwargs...)
    pcapp!(copy(ensure_mat(X)); nsim, kwargs...)
end

function pcapp!(X::Matrix; nsim = 2000, kwargs...)
    par = recovkwargs(Par, kwargs) 
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    xmeans = Jchemo.colmedspa(X) 
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colmad(X)
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
    ## ~ same for large nsim (~ >= 2000):
    #fsimpp = simppsph  
    for a = 1:nlv
        ## For simpphub: the nb. columns of zP can be variable (max = n + A(n, 2))
        zP = fsimpp(X; nsim)  
        zT = X * zP 
        zobj = colmad(zT)
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

