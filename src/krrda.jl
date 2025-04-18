"""
    krrda(; kwargs...)
    krrda(X, y; kwargs...)
    krrda(X, y, weights::Weight; kwargs...)
Discrimination based on kernel ridge regression (KRR-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments: 
* `lb` : Ridge regularization parameter "lambda".
* `kern` : Type of kernel used to compute the Gram matrices. Possible values are: `:krbf`, `:kpol`. See respective 
    functions `krbf` and `kpol` for their keyword arguments.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `scal` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.

Same as function `rrda` (RR-DA) except that a kernel RR (function `krr`), instead of a RR (function `rr`), 
is run on the Y-dummy table. 

## Examples
```julia
using Jchemo, JchemoData, JLD2
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2")
@load db dat
@names dat
X = dat.X
Y = dat.Y
n = nro(X) 
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
(ntot = n, ntrain, ntest)
tab(ytrain)
tab(ytest)

lb = 1e-5
kern = :krbf ; gamma = .001 
scal = true
model = krrda(; lb, kern, gamma, scal) 
fit!(model, Xtrain, ytrain)
@names model
@names model.fitm
fitm = model.fitm ;
fitm.lev
fitm.ni

coef(fitm.fitm)

res = predict(model, Xtest) ;
@names res
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest; lb = [.1, .001]).pred
```
""" 
krrda(; kwargs...) = JchemoModel(krrda, nothing, kwargs)

function krrda(X, y; kwargs...)
    par = recovkw(ParKrrda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    krrda(X, y, weights; kwargs...)
end

function krrda(X, y, weights::Weight; kwargs...)  
    par = recovkw(ParKrrda, kwargs).par
    res = dummy(y)
    ni = tab(y).vals
    fitm = krr(X, res.Y, weights; kwargs...)
    Rrda(fitm, res.lev, ni, par)
end


