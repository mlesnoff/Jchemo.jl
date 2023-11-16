"""
    splslda(X, y, weights = ones(nro(X)); nlv, 
        meth = :soft, delta = 0, nvar = nco(X), 
        prior = :unif, scal::Bool = false)
Sparse PLS-LDA.
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights of the observations. Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `meth`: Method used for the thresholding. Possible values
    are :soft (default), :mix or :hard. See thereafter.
* `delta` : Range for the thresholding (see function `soft`)
    on the loadings standardized to their maximal absolute value.
    Must ∈ [0, 1]. Only used if `meth = :soft.
* `nvar` : Nb. variables (`X`-columns) selected for each 
    LV. Can be a single integer (same nb. variables
    for each LV), or a vector of length `nlv`.
    Only used if `meth = :mix` or `meth = :hard`. 
* `prior` : Type of prior probabilities for class membership.
    Possible values are: :unif (uniform; default), :prop (proportional).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Same as function `plslda` (PLS-LDA) except that sparse PLSR (function 
`splskern`) is run on the Y-dummy table instead of a PLSR (function `plskern`). 

See `?splskern` and `?plslda.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "forages2.jld2") 
@load db dat
pnames(dat)

X = dat.X 
Y = dat.Y 
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]

tab(ytrain)
tab(ytest)

nlv = 15
delta = .8
nvar = 2
scal = false
#scal = true
meth = :soft
#meth = :mix
#meth = :hard
fm = splslda(Xtrain, ytrain; nlv = nlv,
    meth = meth, delta = delta, nvar = nvar,
    scal = scal) ;
pnames(fm)
pnames(fm.fm)
zfm = fm.fm.fmpls ;
zfm.sellv
zfm.sel
res = Jchemo.predict(fm, Xtest)
res.posterior
err(res.pred, ytest)
confusion(res.pred, ytest).cnt

nlv = 1:30 
pars = mpar(meth = [:mix], nvar = [1; 5; 10; 20], 
    scal = [false])
res = gridscorelv(Xtrain, ytrain, Xtest, ytest; 
    score = err, fun = splslda, pars = pars, nlv = nlv)
typ = string.("nvar=", res.nvar)
plotgrid(res.nlv, res.y1, typ; step = 2,
    xlabel = "Nb. LVs", ylabel = "ERR").f
```
""" 
function splslda(X, y, weights = ones(nro(X)); nlv, 
        meth = :soft, delta = 0, nvar = nco(X), 
        prior = :unif, scal::Bool = false)
    res = dummy(y)
    ni = tab(y).vals
    fmpls = splskern(X, res.Y, weights; nlv = nlv, 
        meth = meth, delta = delta, nvar = nvar, 
        scal = scal)
    fmda = list(nlv)
    @inbounds for i = 1:nlv
        fmda[i] = lda(fmpls.T[:, 1:i], y, weights; prior = prior)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Plslda(fm, res.lev, ni)
end






