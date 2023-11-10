"""
    splskern(X, Y, weights = ones(nro(X)); nlv,
        meth = :soft, delta = 0, nvar = nco(X), 
        scal::Bool = false)
    splskern!(X, Y, weights = ones(nro(X)); nlv,
        meth = :soft, delta = 0, nvar = nco(X), 
        scal::Bool = false)
Sparse PLSR (Shen & Huang 2008).
* `X` : X-data (n, p). 
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. latent variables (LVs).
* `meth`: Method used for the thresholding. Possible values
    are :soft (default), :mix or :hard. See thereafter.
* `delta` : Range for the thresholding (see function `soft`)
    on the loadings standardized to their maximal absolute value.
    Must ∈ [0, 1]. Only used if `meth = :soft.
* `nvar` : Nb. variables (`X`-columns) selected for each 
    LV. Can be a single integer (same nb. variables
    for each LV), or a vector of length `nlv`.
    Only used if `meth = :mix` or `meth = :hard`.   
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

Sparse partial least squares regression (Lê Cao et al. 2008), with 
the fast "improved kernel algorithm #1" of Dayal & McGregor (1997). 
In the present version, the sparseness only concerns `X` (not `Y`). 

Function `splskern' provides three methods of thresholding to compute 
the sparse `X`-loading weights w, see `?spca' for description (same 
principles). The case `meth = :mix` returns the same results as function 
spls of the R package mixOmics in regression mode (and with no sparseness 
on `Y`).

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
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlv = 15
fm = splskern(Xtrain, ytrain; nlv = nlv,
    meth = :mix, nvar = 5) ;
pnames(fm)
fm.T
fm.W
fm.P
fm.sellv
fm.sel

zcoef = Jchemo.coef(fm)
zcoef.int
zcoef.B
Jchemo.coef(fm; nlv = 7).B

Jchemo.transform(fm, Xtest)
Jchemo.transform(fm, Xtest; nlv = 7)

res = Jchemo.predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

res = Jchemo.predict(fm, Xtest; nlv = 1:2)
res.pred[1]
res.pred[2]

res = summary(fm, Xtrain) ;
pnames(res)
z = res.explvarx
lines(z.nlv, z.cumpvar,
    axis = (xlabel = "Nb. LVs", ylabel = "Prop. Explained X-Variance"))
```
""" 
function splskern(X, Y, weights = ones(nro(X)); nlv, 
        meth = :soft, delta = 0, nvar = nco(X), scal::Bool = false)
    splskern!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv, 
        meth = meth, delta = delta, nvar = nvar, scal = scal)
end

function splskern!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        meth = :soft, delta = 0, nvar = nco(X), scal::Bool = false)
    @assert in([:soft; :mix; :hard])(meth) "Wrong value for argument 'meth'."
    @assert 0 <= delta <= 1 "Argument 'delta' must ∈ [0, 1]." 
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, nlv)
    length(nvar) == 1 ? nvar = repeat([nvar], nlv) : nothing
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)  
    xscales = ones(eltype(X), p)
    yscales = ones(eltype(Y), q)
    if par.scal 
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
    absw_stand = copy(zp)
    theta = copy(zp)
    r   = copy(zp)
    c   = similar(X, q)
    tmp = similar(XtY) # = XtY_approx
    sellv = list(nlv, Vector{Int})
    @inbounds for a = 1:nlv
        if q == 1
            w .= vcol(XtY, 1)
            absw .= abs.(w)
            if meth == :soft
                absw_max = maximum(absw)
                absw_stand .= absw / absw_max
                theta .= max.(0, absw_stand .- delta) 
                w .= sign.(w) .* theta * absw_max 
            elseif meth == :mix
                nrm = p - nvar[a]
                if nrm > 0
                    sel = sortperm(absw; rev = true)[1:nvar[a]]
                    wmax = w[sel]
                    w .= zeros(p)
                    w[sel] .= wmax
                    delta = maximum(sort(absw)[1:nrm])
                    w .= soft.(w, delta)
                end
            elseif meth == :hard
                sel = sortperm(absw; rev = true)[1:nvar[a]]
                wmax = w[sel]
                w .= zeros(p)
                w[sel] .= wmax
            end
            ## End
            w ./= norm(w)
        else
            if meth == :soft
                w .= snipals(XtY'; delta = delta[a]).v
            elseif meth == :mix
                w .= snipalsmix(XtY'; nvar = nvar[a]).v
            elseif meth == :hard
                w .= snipalsh(XtY'; nvar = nvar[a]).v
            end
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
     Splsr(T, P, R, W, C, TT, xmeans, xscales, ymeans, 
         yscales, weights, nothing, sellv, sel)
end



