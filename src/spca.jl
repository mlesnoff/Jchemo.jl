struct Spca1
    T::Array{Float64} 
    P::Array{Float64}
    sv::Vector{Float64}
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    weights::Vector{Float64}
    niter::Union{Vector{Int64}, Nothing}
    sellv::Vector{Vector{Int64}}
    sel::Vector{Int64}
end

function spca(X, weights = ones(nro(X)); nlv,
        meth = "soft", nvar = nco(X), delta = 0, 
        tol = sqrt(eps(1.)), maxit = 200, scal::Bool = false)
    spca!(copy(ensure_mat(X)), weights; nlv = nlv,
        meth = meth, nvar = nvar, delta = delta, 
        tol = tol, maxit = maxit, scal = scal)
end

function spca!(X::Matrix, weights = ones(nro(X)); nlv, 
        meth = "soft", nvar = nco(X), delta = 0, 
        tol = sqrt(eps(1.)), maxit = 200, scal::Bool = false)
    n, p = size(X)
    nlv = min(nlv, n, p)
    length(nvar) == 1 ? nvar = repeat([nvar], nlv) : nothing
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    xscales = ones(p)
    if scal 
        xscales .= colstd(X, weights)
        cscale!(X, xmeans, xscales)
    else
        center!(X, xmeans)
    end
    sqrtw = sqrt.(weights)
    X .= Diagonal(sqrtw) * X
    t = similar(X, n)
    T = similar(X, n, nlv)
    P = similar(X, p, nlv)
    sv = similar(X, nlv)
    niter = list(nlv, Int64)
    sellv = list(nlv, Vector{Int64})
    for a = 1:nlv
        if meth == "soft"
            res = snipals(X; 
                delta = delta, tol = tol, maxit = maxit)
        elseif meth == "mix"
            res = snipalsmix(X; 
                nvar = nvar[a], tol = tol, maxit = maxit)
        elseif meth == "hard"
            res = snipalsh(X; 
                nvar = nvar[a], tol = tol, maxit = maxit)
        end
        t .= res.t   # = PC = X_defl * v    
        tt = dot(t, t)
        X .-= t * t' * X / tt        
        sv[a] = norm(t)
        T[:, a] .= t ./ sqrtw
        P[:, a] .= res.v           
        niter[a] = res.niter
        sellv[a] = findall(abs.(res.v) .> 0)
    end    
    sel = unique(reduce(vcat, sellv))
    Spca1(T, P, sv, xmeans, xscales, weights, niter,
        sellv, sel) 
end

function Base.summary(object::Spca1, X::Union{Matrix, DataFrame})
    X = ensure_mat(X)
    nlv = size(object.T, 2)
    D = Diagonal(object.weights)
    X = cscale(X, object.xmeans, object.xscales)
    sstot = sum(colnorm(X, object.weights).^2)   # = tr(X' * D * X)
    ## Proportion of variance of X explained by each t
    A = X' * D * object.T
    ss = diag(A' * A) ./ diag(object.T' * D * object.T)
    pvar = ss / sstot 
    cumpvar = cumsum(pvar)
    explvarx = DataFrame(lv = 1:nlv, pvar = pvar, 
        cumpvar = cumpvar)
    (explvarx = explvarx, )
end

