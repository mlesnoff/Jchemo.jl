struct Spca2
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
    b = similar(X, 1, p)
    beta = similar(X, p, nlv)
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
        b .= t' * X / tt           
        X .-= t * b        
        sv[a] = norm(t)
        T[:, a] .= t ./ sqrtw
        P[:, a] .= res.v
        beta[:, a] .= vec(b)
        niter[a] = res.niter
        sellv[a] = findall(abs.(res.v) .> 0)
    end    
    sel = unique(reduce(vcat, sellv))
    Spca2(T, P, sv, beta, xmeans, xscales, weights, niter,
        sellv, sel) 
end

""" 
    transform(object::Spca2, X; nlv = nothing)
    Compute principal components (PCs = scores T) from a fitted model and X-data.
* `object` : The fitted model.
* `X` : X-data for which PCs are computed.
* `nlv` : Nb. PCs to compute.
""" 
function transform(object::Spca2, X; nlv = nothing)
    X = ensure_mat(X)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    zX = cscale(X, fm.xmeans, fm.xscales)
    T = zeros(n, nlv)
    for a = 1:nlv
        tnew = zX * object.P[:, a]
        T[:, a] .= tnew
        zX .= zX .- tnew * object.beta[:, a]'
    end
    T 
end

"""
    summary(object::Spca2, X::Union{Matrix, DataFrame})
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Spca2, X::Union{Matrix, DataFrame})
    X = ensure_mat(X)
    nlv = size(object.T, 2)
    D = Diagonal(object.weights)
    X = cscale(X, object.xmeans, object.xscales)
    sstot = sum(colnorm(X, object.weights).^2)   # = tr(X' * D * X) = frob(X, weights)^2    
    ## Proportion of variance of X explained by each column of T
    ## ss = diag(T' * D * X * X' * D * T) ./ diag(T' * D * T)
    A = X' * D * object.T    
    ss = diag(A' * A) ./ diag(object.T' * D * object.T)
    pvar = ss / sstot 
    cumpvar = cumsum(pvar)
    zrd = vec(rd(X, object.T, object.weights))
    explvarx = DataFrame(lv = 1:nlv, rd = zrd, 
        pvar = pvar, cumpvar = cumpvar)
    ## Adjusted variance (Shen & Hunag 2008 section 2.3)
    zX = sqrt.(D) * X
    ss = zeros(nlv)
    for a = 1:nlv
        P = object.P[:, 1:a]
        Xadj = zX * P * inv(P' * P) * P'
        ss[a] = sum(Xadj.^2)
    end
    cumpvar = ss / sstot
    pvar = [cumpvar[1]; diff(cumpvar)]
    explvarx_adj = DataFrame(lv = 1:nlv, 
        pvar = pvar, cumpvar = cumpvar)
    ## End
    (explvarx = explvarx, explvarx_adj)
end

