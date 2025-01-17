"""
    spca(; kwargs...)
    spca(X; kwargs...)
    spca(X, weights::Weight; kwargs...)
    spca!(X::Matrix, weights::Weight; kwargs...)
Sparse PCA (Shen & Huang 2008).
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. principal components (PCs).
* `meth` : Method used for the sparse thresholding. 
    Possible values are: `:soft`, `:hard`. See thereafter.
* `nvar` : Nb. variables (`X`-columns) selected for each principal
    component (PC). Can be a single integer (i.e. same nb. 
    of variables for each PC), or a vector of length `nlv`.   
* `tol` : Tolerance value for stopping the Nipals iterations.
* `maxit` : Maximum nb. of Nipals iterations.
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

sPCA-rSVD algorithm (regularized low rank matrix approximation) of 
Shen & Huang 2008. 

The algorithm computes each loadings vector iteratively, by alternating 
least squares regressions (Nipals) including a step of thresholding. Function 
`spca` provides thresholding methods '1' and '2' (`:soft` and `:hard`) reported 
in Shen & Huang 2008 Lemma 2:
* The tuning parameter used by Shen & Huang 2008 is the number of null elements 
    in the loadings vector, referred to as degree of sparsity. Conversely, the 
    present function `spca` uses the number of non-zero elements (`nvar`), 
    equal to p - degree of sparsity.
* See the code of function `snipals_shen` for details on how is computed 
    the cutoff 'lambda' used inside the thresholding function (Shen & Huang 2008), 
    given a value for `nvar`. Differences from other softwares may occur 
    when there are tied values in the loadings vector (depending on the choices 
    of method used to compute quantiles).

To deflate matrix `X` after a given PC, the present function `spca` does a 
regression of the `X`-columns on the score vector `t`. When `meth = :soft`, the function 
gives the same result as function `spca` of the R package `mixOmics` (except possibly 
when there are many tied values in the loadings vectors, which is not usual). 

The computed sparse loadings vectors (`V`-columns) are in general non orthogonal. 
Therefore, there is no a unique decomposition of the variance of `X` such as in PCA. 
Function `summary` returns the following objects:
* `explvarx`: The proportion of variance of `X` explained 
    by each column `t` of `T`, computed by regressing `X` 
    on `t` (such as what is usually done in PLS).
* `explvarx_v`: Adjusted explained variance proposed by 
    Shen & Huang 2008 section 2.3, that uses regressions 
    of the `X`-rows on the loadings spaces `V`.     

## References
Kim-Anh Lê Cao, Florian Rohart, Ignacio Gonzalez, Sebastien 
Dejean with key contributors Benoit Gautier, Francois Bartolo, 
contributions from Pierre Monget, Jeff Coquery, FangZou Yao 
and Benoit Liquet. (2016). mixOmics: Omics Data Integration 
Project. R package version 6.1.1. 
https://www.bioconductor.org/packages/release/bioc/html/mixOmics.html

Shen, H., Huang, J.Z., 2008. Sparse principal component 
analysis via regularized low rank matrix approximation. 
Journal of Multivariate Analysis 99, 1015–1034. 
https://doi.org/10.1016/j.jmva.2007.06.007

## Examples
```julia
using Jchemo, JchemoData, JLD2 
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
@load db dat
pnames(dat)
@head dat.X
X = dat.X[:, 1:4]
n = nro(X)
ntest = 30
s = samprand(n, ntest) 
Xtrain = X[s.train, :]
Xtest = X[s.test, :]

nlv = 3 
meth = :soft
#meth = :hard
nvar = 2
model = spca(; nlv, meth, nvar) ;
fit!(model, Xtrain) 
fitm = model.fitm ;
pnames(fitm)
fitm.niter
fitm.sellv 
fitm.sel
V = fitm.V
V' * V
@head T = fitm.T
T' * T
@head transf(model, Xtrain)

@head Ttest = transf(fitm, Xtest)

res = summary(model, Xtrain) ;
res.explvarx
res.explvarx_v
```
"""
spca(; kwargs...) = JchemoModel(spca, nothing, kwargs)

function spca(X; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    spca(X, weights; kwargs...)
end

function spca(X, weights::Weight; kwargs...)
    spca!(copy(ensure_mat(X)), weights; kwargs...)
end

function spca!(X::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParSpca, kwargs).par
    @assert in([:shen; :post])(par.algo) "Wrong value for argument 'algo'."
    @assert in([:soft; :hard])(par.meth) "Wrong value for argument 'meth'."
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
    if par.algo == :shen 
        snipals = snipals_shen
    elseif par.algo == :mix 
        snipals = Jchemo.snipals_mix
    elseif par.algo == :post 
        snipals = Jchemo.snipals_post
    end
    nvar = par.nvar
    length(nvar) == 1 ? nvar = repeat([nvar], nlv) : nothing
    xmeans = colmean(X, weights) 
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    sqrtw = sqrt.(weights.w)
    fweight!(X, sqrtw)
    T = similar(X, n, nlv)
    V = similar(X, p, nlv)
    sv = similar(X, nlv)
    niter = list(Int, nlv)
    b = similar(X, 1, p)
    beta = similar(X, p, nlv)
    sellv = list(Vector{Int}, nlv)
    for a = 1:nlv
        res = snipals(X; meth = par.meth, nvar = nvar[a], tol = par.tol, 
            maxit = par.maxit)
        ## Deflation by regression of X-columns on t
        ## same as in plsnipals      
        tt = dot(res.t, res.t)
        b .= res.t' * X / tt           
        X .-= res.t * b   #  = X - t * t' X / tt
        ## End        
        sv[a] = normv(res.t)
        T[:, a] .= res.t ./ sqrtw
        V[:, a] .= res.v
        beta[:, a] .= vec(b)
        niter[a] = res.niter
        sellv[a] = findall(abs.(res.v) .> 0)
    end    
    sel = unique(reduce(vcat, sellv))
    Spca(T, V, sv, beta, xmeans, xscales, weights, niter, sellv, sel, par) 
end

""" 
    transf(object::Spca, X; nlv = nothing)
    Compute principal components (PCs = scores T) from a 
        fitted model and X-data.
* `object` : The fitted model.
* `X` : X-data for which PCs are computed.
* `nlv` : Nb. PCs to compute.
""" 
function transf(object::Spca, X; nlv = nothing)
    X = ensure_mat(X)
    m = nro(X)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    zX = fcscale(X, object.xmeans, object.xscales)
    T = similar(X, m, nlv)
    for a = 1:nlv
        T[:, a] .= zX * vcol(object.V, a)
        zX .-= vcol(T, a) * vcol(object.beta, a)' 
    end
    T 
end

"""
    summary(object::Spca, X)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Spca, X)
    X = ensure_mat(X)
    nlv = nco(object.T)
    weights = object.weights
    sqrtw = sqrt.(weights.w)
    X = fcscale(X, object.xmeans, object.xscales)
    sstot = frob2(X, weights)
    TT = fweight(object.T.^2, weights.w) 
    tt = colsum(TT) 
    ## Proportion of variance of X explained by each column of T
    A = X' * fweight(object.T, weights.w)
    ss = colnorm(A).^2 ./ colnorm(object.T, object.weights).^2
    ## = diag(T' * D * X * X' * D * T) ./ diag(T' * D * T)
    ## = diag(A' * A) ./ diag(object.T' * D * object.T)
    pvar = ss / sstot 
    cumpvar = cumsum(pvar)
    zrd = vec(rd(X, object.T, weights))
    explvarx = DataFrame(nlv = 1:nlv, rd = zrd, pvar = pvar, cumpvar = cumpvar)
    ## Adjusted variance and CPEV (cumulative percentage of explained variance)
    ## of Shen & Huang 2008 section 2.3
    zX = fweight(X, sqrtw)
    ss = zeros(nlv)
    for a = 1:nlv
        V = vcol(object.V, 1:a)
        Xadj = zX * V * inv(V' * V) * V'
        ss[a] = sum(Xadj.^2)
    end
    cumpvar = ss / sstot
    pvar = [cumpvar[1]; diff(cumpvar)]
    explvarx_v = DataFrame(nlv = 1:nlv, pvar = pvar, cumpvar = cumpvar)
    ## End
    nam = string.("lv", 1:nlv)
    contr_ind = DataFrame(fscale(TT, tt), nam)
    contr_var = DataFrame(object.V.^2, nam)
    ## Should be ok 
    C = X' * fweight(fscale(object.T, sqrt.(tt)), weights.w) 
    coord_var = DataFrame(C, nam)
    ## End
    cor_circle = DataFrame(corm(X, object.T, weights), nam)
    (explvarx = explvarx, explvarx_v, contr_ind, contr_var, coord_var, cor_circle)
end

