"""
    splskern(X, Y, weights = ones(nro(X)); nlv,
        meth = "soft", nvar = nco(X), delta = 0, 
        tol = sqrt(eps(1.)), maxit = 200, scal::Bool = false)
    splskern!(X, Y, weights = ones(nro(X)); nlv,
        meth = "soft", nvar = nco(X), delta = 0, 
        tol = sqrt(eps(1.)), maxit = 200, scal::Bool = false)
Sparse PLSR (Shen & Huang 2008).
* `X` : X-data (n, p). 
* `Y` : Y-data (n, q).
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

Sparse partial least squares regression (Lê CAo et al. 2008). 

Function `splskern' uses the fast PLSR "improved kernel algorithm #1"
of Dayal & McGregor (1997). The sparsity is only for `X`. 

Function `splskern' provides three methods of thresholding to compute 
the sparse loading-weights w:

* `meth = "soft"`: Soft thresholding of standardized loadings. 
    Noting w the loading vector, at each step, abs(w) is standardized to 
    its maximal component (= max{abs(w[i]), i = 1..p}). The soft-thresholding 
    function (see function `soft`) is applied to this standardized vector, 
    with the constant `delta` ∈ [0, 1]. This returns the sparse vector 
    theta. Vector w is multiplied term-by-term by vector theta, which
    finally gives the sparse loadings.

* `meth = "mix"`: Method used in function `splskern` of the R package `mixOmics`.
    For each PC, a number of `X`-variables showing the largest 
    values in vector abs(w) are selected. Then a soft-thresholding is 
    applied to the corresponding selected loadings. Range `delta` is 
    automatically (internally) set to the maximal value of the components 
    of abs(w) corresponding to variables removed from the selection.  

* `meth = "hard"`: For each PC, a number of `X-variables showing 
    the largest values in vector abs(w) are selected.

Since the resulting sparse loadings vectors (`P`-columns) are in general 
non orthogonal, there is no a unique decomposition of the variance of `X` 
such as in PCA. Function `summary` returns the following objects:
* `explvarx`: The proportion of variance of `X` explained by each column 
    t of `T`, computed by regressing `X` on t (such as what is done in PLS).
* `explvarx_adj`: Adjusted explained variance proposed by 
    Shen & Huang 2008 section 2.3.    

## References
Cao, K.-A.L., Rossouw, D., Robert-Granié, C., Besse, P., 2008. A Sparse PLS 
for Variable Selection when Integrating Omics Data. Statistical Applications 
in Genetics and Molecular Biology 7. https://doi.org/10.2202/1544-6115.1390

Kim-Anh Lê Cao, Florian Rohart, Ignacio Gonzalez, Sebastien Dejean with key 
contributors Benoit Gautier, Francois Bartolo, contributions from Pierre Monget, 
Jeff Coquery, FangZou Yao and Benoit Liquet. (2016). 
mixOmics: Omics Data Integration Project. R package version 6.1.1. 
https://CRAN.R-project.org/package=mixOmics

https://www.bioconductor.org/packages/release/bioc/html/mixOmics.html

Dayal, B.S., MacGregor, J.F., 1997. Improved PLS algorithms. 
Journal of Chemometrics 11, 73-85.

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

tol = 1e-15
nlv = 3 
scal = false
#scal = true
meth = "soft"
#meth = "mix"
#meth = "hard"
nvar = 2 ; delta = .4
fm = splskern(Xtrain; nlv = nlv, 
    meth = meth, nvar = nvar, delta = delta, 
    tol = tol, scal = scal) ;
fm.niter
fm.sellv 
fm.sel
fm.P
fm.P' * fm.P
head(fm.T)

Ttest = Jchemo.transform(fm, Xtest)

res = Jchemo.summary(fm, Xtrain) ;
res.explvarx
res.explvarx_adj
```
""" 
function splskern(X, Y, weights = ones(nro(X)); nlv, 
        nvar = nco(X), scal::Bool = false)
    splskern!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; 
        nvar = nvar, nlv = nlv, scal = scal)
end

function splskern!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        nvar = nco(X), scal::Bool = false)
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, nlv)
    length(nvar) == 1 ? nvar = repeat([nvar], nlv) : nothing
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)  
    xscales = ones(p)
    yscales = ones(q)
    if scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        cscale!(X, xmeans, xscales)
        cscale!(Y, ymeans, yscales)
    else
        center!(X, xmeans)
        center!(Y, ymeans)
    end
    D = Diagonal(weights)
    XtY = X' * (D * Y)                   # = Xd' * Y = X' * D * Y  (Xd = D * X   Very costly!!)
    #XtY = X' * (weights .* Y)           # Can create OutOfMemory errors for very large matrices
    # Pre-allocation
    T = similar(X, n, nlv)
    W = similar(X, p, nlv)
    P = copy(W)
    R = copy(P)
    C = similar(X, q, nlv)
    TT = similar(X, nlv)
    t   = similar(X, n)
    dt  = copy(t)   
    zp  = similar(X, p)
    w   = copy(zp)
    absw = copy(zp)
    r   = copy(zp)
    c   = similar(X, q)
    tmp = similar(XtY) # = XtY_approx
    sellv = list(nlv, Vector{Int64})
    @inbounds for a = 1:nlv
        if q == 1
            w .= vcol(XtY, 1)
            ## Sparsity
            nrm = p - nvar[a]
            if nrm > 0
                absw .= abs.(w)
                sel = sortperm(absw; rev = true)[1:nvar[a]]
                wmax = w[sel]
                w .= zeros(p)
                w[sel] .= wmax
                delta = maximum(sort(absw)[1:nrm])
                w .= soft.(w, delta)
            end
            ## End
            w ./= norm(w)
        else
            w .= snipalsmix(XtY'; nvar = nvar[a]).v
        end                                  
        r .= w
        if a > 1
            @inbounds for j = 1:(a - 1)
                r .-= dot(w, vcol(P, j)) .* vcol(R, j)    
            end
        end                   
        mul!(t, X, r)                 # t = X * r
        dt .= weights .* t            # dt = D * t
        tt = dot(t, dt)               # tt = t' * dt = t' * D * t 
        mul!(c, XtY', r)
        c ./= tt                      # c = XtY' * r / tt
        mul!(zp, X', dt)              # zp = (D * X)' * t = X' * (D * t)
        XtY .-= mul!(tmp, zp, c')     # XtY = XtY - zp * c' ; deflation of the kernel matrix 
        P[:, a] .= zp ./ tt           # ==> the metric applied to covariance is applied outside the loop,
        T[:, a] .= t                  # conversely to other algorithms such as nipals
        W[:, a] .= w
        R[:, a] .= r
        C[:, a] .= c
        TT[a] = tt
        sellv[a] = findall(abs.(w) .> 0)
     end
     sel = unique(reduce(vcat, sellv))
     Plsr(T, P, R, W, C, TT, xmeans, xscales, ymeans, 
         yscales, weights, nothing)
end



