"""
    covselr(X, Y, weights = ones(nro(X)); nlv,
        nvar = 1; scal::Bool = false)
    covselr!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        nvar = 1, scal::Bool = false)
Covsel regression.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. latent variables (LVs) to compute.
* `nvar` : Nb. variables (`X`-columns) selected for each 
    latent variable (LV). Default `nvar = 1` corresponds to 
    the Covsel regression (Roger et al. 2011). 
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.

The general principle of Covsel regression (Roger et al. 2011; see also 
Höskuldsson, A., 1992, appendix) is in two steps: 
* A number of variables (`X`-columns) maximizing partial 
covariances are selected as follows: a first variable (maximizing 
squared covariance with `Y`) is selected , `X` is deflated to this variable, 
and a new variable is then selected on the same criterion, etc.
* Then, a MLR is run on the selected variables.

Function `covsel` is an extension of the Covsel regression algorithm: 
* The function uses the fact that Covsel regression can be considered as a 
particular case of PLSR algorithm (Roger et al 2011).  The function is 
an adaptation of function `plskern` where a hard thresholding of 
the loading weigths (w1, w2, ..., wp) was adeded. This generates much faster 
computation times than selecting the usual approach sleceting the variables 
and then fitting a MLR on these variables. 
* Using argument `nvar`, the fonction allows to select more than one variable 
per latent variable (LV) of the adapted PLSR. For each LV, by hard thresholding,
the weights wj (j = 1, ..., p) are set to 0 except the `nvar`largest abs(wj).

When `Y` is multivariate, it is generally recommended to scale the data
(argument `scale`) to give potentially the same importance to each `Y`-column 
in the selection.

## References
Höskuldsson, A., 1992. The H-principle in modelling with applications 
to chemometrics. Chemometrics and Intelligent Laboratory Systems, 
Proceedings of the 2nd Scandinavian Symposium on Chemometrics 14, 
139–153. https://doi.org/10.1016/0169-7439(92)80099-P

Roger, J.M., Palagos, B., Bertrand, D., Fernandez-Ahumada, E., 2011. 
covsel: Variable selection for highly multivariate and multi-response 
calibration: Application to IR spectroscopy. 
Chem. Lab. Int. Syst. 106, 216-223.

Wikipedia
https://en.wikipedia.org/wiki/Partial_correlation

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
fm = covselr(Xtrain, ytrain; nlv = nlv,
    nvar = 1) ;  # Covsel regressin
pnames(fm)
fm.sel
fm.sellv

zcoef = Jchemo.coef(fm)
zcoef.int
zcoef.B

res = Jchemo.predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

res = summary(fm, Xtrain) ;
pnames(res)
z = res.explvarx
lines(z.nlv, z.cumpvar,
    axis = (xlabel = "Nb. LVs", ylabel = "Prop. Explained X-Variance"))
```
""" 
function covselr(X, Y, weights = ones(nro(X)); nlv, 
        nvar = 1, scal::Bool = false)
    covselr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; 
        nlv = nlv, nvar = nvar, scal = scal)
end

function covselr!(X::Matrix, Y::Matrix, weights = ones(nro(X)); 
        nlv, nvar = 1, scal::Bool = false)
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, nlv)
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
    dt  = similar(X, n)   
    zp  = similar(X, p)
    w   = similar(X, p)
    r   = similar(X, p)
    c   = similar(X, q)
    tmp = similar(XtY) # = XtY_approx
    sellv = list(nlv, Vector{Int64})
    zw = copy(w)
    # End
    @inbounds for a = 1:nlv
        if q == 1
            w .= vcol(XtY, 1)
            w ./= norm(w)
        else
            w .= svd(XtY).U[:, 1]
        end
        ## Sparsity
        zw .= abs.(w)
        sellv[a] = sortperm(zw; rev = true)[1:nvar]
        wmax = w[sellv[a]]
        w .= zeros(p)
        w[sellv[a]] .= wmax
        ## End
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
     end
     sel = unique(reduce(vcat, sellv))
     Covselr(T, P, R, W, C, TT, xmeans, xscales, ymeans, 
         yscales, weights, sellv, sel)
end

