"""
    spca(; kwargs...)
    spca(X; kwargs...)
    spca(X, weights::Weight; kwargs...)
    spca!(X::Matrix, weights::Weight; kwargs...)
Sparse PCA by regularized low rank matrix approximation (sPCA-rSVD, Shen & Huang 2008).
* `X` : X-data (n, p). 
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. principal components (PCs).
* `meth` : Method used for the sparse thresholding. Possible values are: `:soft`, `:hard`. See thereafter.
* `nvar` : Nb. variables (`X`-columns) selected for each principal component (PC). Can be a single integer 
    (i.e. same nb. of variables for each PC), or a vector of length `nlv`.   
* `defl` : Type of `X`-matrix deflation, see below.
* `tol` : Tolerance value for stopping the Nipals iterations.
* `maxit` : Maximum nb. of Nipals iterations.
* `scal` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.

sPCA-rSVD algorithm (regularized low rank matrix approximation) of Shen & Huang 2008. 

The algorithm computes each loadings vector iteratively, by alternating 
least squares regressions (Nipals) including a step of thresholding. Function 
`spca` provides thresholding methods '1' and '2' reported in Shen & Huang 2008 
Lemma 2 (`:soft` and `:hard`):
* The tuning parameter used by Shen & Huang 2008 is the number of null elements 
    in the loadings vector, referred to as degree of sparsity. Conversely, the 
    present function `spca` uses the number of non-zero elements (`nvar`), 
    equal to p - degree of sparsity.
* See the code of function `snipals_shen` for details on how is computed 
    the cutoff 'lambda' used inside the thresholding function (Shen & Huang 2008), 
    given a value for `nvar`. Differences from other softwares may occur 
    when there are tied values in the loadings vector (depending on the choices 
    of method used to compute quantiles).

Matrix `X` can be deflated in two ways:
* `defl = :v` : Matrix `X` is deflated by regression of the `X'`-columns on 
  the loadings vector `v`. This is the method proposed by Shen & Huang 2008 
  (see in Theorem A.2 p.1033).
* `defl = :t` : Matrix `X` is deflated by regression of the `X`-columns on 
  the score vector `t`. This is the method used in function `spca` of the 
  R package `mixOmics` (Le Cao et al. 2016).
The method of computation of the % variance explained in X by each PC (returned 
by function `summary`) depends on the type of deflation chosen (see the code).    

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
@names dat
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
@names fitm
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
    @assert in([:v; :t])(par.defl) "Wrong value for argument 'defl'."
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
        ## Deflation
        if par.defl == :v       # regression X' on v (S&H2008 in Th.A.2 p.1033)
            X .-= res.t * res.v'
        elseif par.defl == :t   # Regression X on t (e.g. R mixOmics::spca)
            tt = dot(res.t, res.t)
            b .= res.t' * X / tt           
            X .-= res.t * b   #  = X - t * t' X / tt
        end
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
    defl = object.par.defl
    for a = 1:nlv
        T[:, a] .= zX * vcol(object.V, a)
        ## Deflation
        if defl == :v       
            zX .-= vcol(T, a) * vcol(object.V, a)'
        elseif defl == :t   
            zX .-= vcol(T, a) * vcol(object.beta, a)'
        end
        ## End   
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
    defl = object.par.defl 
    if defl == :v      
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
        explvarx = DataFrame(nlv = 1:nlv, pvar = pvar, cumpvar = cumpvar)
    elseif defl == :t
        ## Proportion of variance of X explained by each column of T 
        A = X' * fweight(object.T, weights.w)
        ss = colnorm(A).^2 ./ colnorm(object.T, object.weights).^2
        ## = diag(T' * D * X * X' * D * T) ./ diag(T' * D * T)
        ## = diag(A' * A) ./ diag(object.T' * D * object.T)
        pvar = ss / sstot 
        cumpvar = cumsum(pvar)
        zrd = vec(rd(X, object.T, weights))
        explvarx = DataFrame(nlv = 1:nlv, rd = zrd, pvar = pvar, cumpvar = cumpvar)
    end
    nam = string.("lv", 1:nlv)
    contr_ind = DataFrame(fscale(TT, tt), nam)
    contr_var = DataFrame(object.V.^2, nam)
    ## Should be ok 
    C = X' * fweight(fscale(object.T, sqrt.(tt)), weights.w) 
    coord_var = DataFrame(C, nam)
    ## End
    cor_circle = DataFrame(corm(X, object.T, weights), nam)
    (explvarx = explvarx, contr_ind, contr_var, coord_var, cor_circle)
end

