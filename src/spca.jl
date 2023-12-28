"""
    spca(; kwargs...)
    spca(X; kwargs...)
    spca(X, weights::Weight; kwargs...)
    spca!(X::Matrix, weights::Weight; 
        kwargs...)
Sparse PCA (Shen & Huang 2008).
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. principal components (PCs).
* `msparse` : Method used for the sparse thresholding. 
    Possible values are: `:soft`, `:mix`, 
    `:hard`. See thereafter.
* `delta` : Only used if `msparse = `:soft`. Range for the 
    thresholding on the loadings (after they are standardized 
    to their maximal absolute value). Must ∈ [0, 1].
    Higher is `delta`, stronger is the thresholding. 
* `nvar` : Only used if `msparse` = `:mix` or `:hard`.
    Nb. variables (`X`-columns) selected for each principal
    component (PC). Can be a single integer (i.e. same nb. 
    of variables for each PC), or a vector of length `nlv`.   
* `tol` : Tolerance value for stopping the iterations.
* `maxit` : Maximum nb. of Nipals iterations.
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

Sparse principal component analysis via regularized low rank 
matrix approximation (Shen & Huang 2008). A Nipals algorithm is used. 
The Function provides three methods of thresholding to compute 
the sparse loadings:
* `msparse = :soft` : Soft thresholding of standardized loadings. 
    Let us note v a given loading vector before thresholding. 
    Vector abs(v) is then standardized to its maximal component 
    (= max{abs(v[i]), i = 1..p}). The soft-thresholding function 
    (see function `soft`) is applied to this standardized vector, 
    with the constant `delta` ∈ [0, 1]. This returns the sparse 
    vector `theta`. Vector v is multiplied term-by-term by this vector
    `theta`, which finally gives the sparse loadings.

* `msparse = :mix`: Method used in function `spca` of the R 
    package `mixOmics` (Le Cao et al.). For each PC, the `nvar` 
    `X`-variables showing the largest values in vector abs(v) 
    are selected. Then a soft-thresholding is applied to the 
    corresponding selected loadings. Range `delta` is automatically
    (internally) set equal to the maximal value of the components 
    of abs(v) corresponding to variables removed from the selection.  

* `msparse = :hard`: For each PC, the `nvar` `X`-variables showing 
    the largest values in vector abs(v) are selected.

The case `msparse = :mix` returns the same results as function 
`spca` of the R package mixOmics.

**Note:** The resulting sparse loadings vectors (`P`-columns) 
are in general non orthogonal. Therefore, there is no a unique 
decomposition of the variance of `X` such as in PCA. 
Function `summary` returns the following objects:
* `explvarx`: The proportion of variance of `X` explained 
    by each column t of `T`, computed by regressing `X` 
    on t (such as what is done in PLS).
* `explvarx_adj`: Adjusted explained variance proposed by 
    Shen & Huang 2008 section 2.3.    

## References
Kim-Anh Le Cao, Florian Rohart, Ignacio Gonzalez, Sebastien Dejean 
with key contributors Benoit Gautier, Francois Bartolo, contributions 
from Pierre Monget, Jeff Coquery, FangZou Yao and Benoit Liquet. (2016). 
mixOmics: Omics Data Integration Project. R package version 6.1.1. 
https://CRAN.R-project.org/package=mixOmics

https://www.bioconductor.org/packages/release/bioc/html/mixOmics.html

Shen, H., Huang, J.Z., 2008. Sparse principal component analysis via 
regularized low rank matrix approximation. Journal of Multivariate Analysis 
99, 1015–1034. https://doi.org/10.1016/j.jmva.2007.06.007

## Examples
```julia
using JchemoData, JLD2 
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
msparse = :mix ; nvar = 2
#msparse = :hard ; nvar = 2
scal = false
mod = spca(; nlv, msparse, 
    nvar, scal) ;
fit!(mod, Xtrain) 
fm = mod.fm ;
pnames(fm)
fm.niter
fm.sellv 
fm.sel
fm.P
fm.P' * fm.P
@head T = fm.T
@head transf(mod, Xtrain)

@head Ttest = transf(fm, Xtest)

res = summary(mod, Xtrain) ;
res.explvarx
res.explvarx_adj

nlv = 3 
msparse = :soft ; delta = .4 
mod = spca(; nlv, msparse, 
    delta) ;
fit!(mod, Xtrain) 
mod.fm.P
```
""" 
function spca(X; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    spca(X, weights; kwargs...)
end

function spca(X, weights::Weight; kwargs...)
    spca!(copy(ensure_mat(X)), weights; 
        kwargs...)
end

function spca!(X::Matrix, weights::Weight; 
        kwargs...)
    par = recovkwargs(Par, kwargs) 
    @assert in([:hard ; :soft ; :mix])(par.msparse) "Wrong value for argument 'msparse'."
    @assert 0 <= par.delta <= 1 "Argument 'delta' must ∈ [0, 1]."
    Q = eltype(X)
    n, p = size(X)
    nlv = min(par.nlv, n, p)
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
    X .= Diagonal(sqrtw) * X
    t = similar(X, n)
    T = similar(X, n, nlv)
    P = similar(X, p, nlv)
    sv = similar(X, nlv)
    niter = list(Int, nlv)
    b = similar(X, 1, p)
    beta = similar(X, p, nlv)
    sellv = list(Vector{Int}, nlv)
    for a = 1:nlv
        if par.msparse == :soft
            res = snipals(X; kwargs...)
        else
            par.nvar = nvar[a]
            if par.msparse == :hard
                res = snipalsh(X; kwargs...)
            elseif par.msparse == :mix
                res = snipalsmix(X; kwargs...)
            end
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
        sellv, sel, kwargs, par) 
end

""" 
    transf(object::Spca, X; nlv = nothing)
    Compute principal components (PCs = scores T) from a fitted model and X-data.
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
    t = similar(X, m)
    for a = 1:nlv
        t .= zX * object.P[:, a]
        T[:, a] .= t
        zX .-= t * vcol(object.beta, a)' 
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
    D = Diagonal(object.weights.w)
    X = fcscale(X, object.xmeans, object.xscales)
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
    ## Adjusted variance (Shen & Huang 2008 section 2.3)
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

