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

"""
    spca(X, weights = ones(nro(X)); nlv,
        meth = "soft", nvar = nco(X), delta = 0, 
        tol = sqrt(eps(1.)), maxit = 200, scal::Bool = false)
    spca!(X, weights = ones(nro(X)); nlv,
        meth = "soft", nvar = nco(X), delta = 0, 
        tol = sqrt(eps(1.)), maxit = 200, scal::Bool = false)
Sparse PCA (Shen & Huang 2008).
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. principal components (PCs).
* `meth`: Method used for the thresholding. Possible values
    are "soft" (default), "mix" or "hard". See thereafter.
* `nvar` : Nb. variables (`X`-columns) selected for each 
    PC. Can be a single integer (same nb. variables
    for each PC), or a vector of length `nlv`.
    Only used if `meth = "mix"` or `meth = "hard"`.   
* `delta` : Range for the thresholding (see function `soft`)
    on the loadings standardized to their maximal absolute value.
    Must be within [0, 1]. Only used if `meth = "soft".
* `tol` : Tolerance value for stopping the iterations.
* `maxit` : Maximum nb. iterations.
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

Sparse principal component analysis via regularized low rank 
matrix approximation (Shen & Huang 2008). A Nipals algorithm is used. 

Function `spca' provides three methods of thresholding to compute 
sparse loadings:

* `meth = "soft"`: Soft thresholding on standardized loadings. 
    Noting v the loading vector, at each step, abs(v) is standardized to 
    its maximal value. The soft-thresholding function 
    (see function `soft`) is applied to this standardized vector, 
    with the constant `delta` ∈ [0, 1], which returns the sparse vector 
    theta. Vector v is finally multiplied term-by-term by vector theta
    that gives the sparse loadings.

* `meth = "mix"`: Method used in function `spca` of the R package `mixOmics`.
    For each PC, a number of variables (loadings) showing the largest 
    values in vector abs(v) are selected. Then a soft-thresholding is 
    applied to this selected loadings, whith `delta` set to the maximal value
    of the elements of abs(v) that were removed from the selection.  

* `meth = "hard"`: For each PC, a number of variables (loadings) showing 
    the largest values in vector abs(v) are selected.

Since the sparse loadings vectors (`P`-columns) are in general 
non orthogonal, there is no a unique variance decomposition of `X` such 
as in PCA. Function `summary` returns the following objects:
* `explvarx`: The proportion of variance of `X` explained by each column 
    t of `T` is computed by regressing `X` on t (such as what is done in PLS).
* `explvarx_adj`: Adjusted explained variance proposed by 
    Shen & Huang 2008 section 2.3.    

## References
Kim-Anh Le Cao, Florian Rohart, Ignacio Gonzalez, Sebastien Dejean with key 
contributors Benoit Gautier, Francois Bartolo, contributions from Pierre Monget, 
Jeff Coquery, FangZou Yao and Benoit Liquet. (2016). 
mixOmics: Omics Data Integration Project. R package version 6.1.1. 
https://CRAN.R-project.org/package=mixOmics

https://www.bioconductor.org/packages/release/bioc/html/mixOmics.html

Shen, H., Huang, J.Z., 2008. Sparse principal component analysis via 
regularized low rank matrix approximation. Journal of Multivariate Analysis 
99, 1015–1034. https://doi.org/10.1016/j.jmva.2007.06.007

## Examples
```julia
using JchemoData, JLD2, CairoMakie, StatsBase
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
@load db dat
pnames(dat)
summ(dat.X)

X = dat.X[:, 1:4]
n = nro(X)

ntrain = 120
s = sample(1:n, ntrain; replace = false) 
Xtrain = X[s, :]
Xtest = rmrow(X, s)

nlv = 3
fm = pcasvd(Xtrain; nlv = nlv) ;
#fm = pcaeigen(Xtrain; nlv = nlv) ;
#fm = pcaeigenk(Xtrain; nlv = nlv) ;
#fm = pcanipals(Xtrain; nlv = nlv) ;
pnames(fm)
fm.T
fm.T' * fm.T
fm.P' * fm.P

Jchemo.transform(fm, Xtest)

res = Base.summary(fm, Xtrain) ;
pnames(res)
res.explvarx
res.contr_var
res.coord_var
res.cor_circle
```
""" 
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
        t .= res.t      
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
    Spca(T, P, sv, beta, xmeans, xscales, weights, niter,
        sellv, sel) 
end

""" 
    transform(object::Spca, X; nlv = nothing)
    Compute principal components (PCs = scores T) from a fitted model and X-data.
* `object` : The fitted model.
* `X` : X-data for which PCs are computed.
* `nlv` : Nb. PCs to compute.
""" 
function transform(object::Spca, X; nlv = nothing)
    X = ensure_mat(X)
    m, a = size(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    zX = cscale(X, object.xmeans, object.xscales)
    T = similar(X, m, nlv)
    t = similar(X, m)
    for a = 1:nlv
        t .= zX * object.P[:, a]
        T[:, a] .= t
        zX .-= t * vcol(object.beta, a)'  #object.beta[:, a]'
    end
    T 
end

"""
    summary(object::Spca, X::Union{Matrix, DataFrame})
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Spca, X::Union{Matrix, DataFrame})
    X = ensure_mat(X)
    nlv = nco(object.T)
    D = Diagonal(object.weights)
    X = cscale(X, object.xmeans, object.xscales)
    ## (||X||_D)^2 = tr(X' * D * X) = frob(X, weights)^2
    sstot = sum(colnorm(X, object.weights).^2)    
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

