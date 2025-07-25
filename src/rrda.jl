"""
    rrda(; kwargs...)
    rrda(X, y; kwargs...)
    rrda(X, y, weights::Weight; kwargs...)
Discrimination based on ridge regression (RR-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`). 
Keyword arguments: 
* `lb` : Ridge regularization parameter "lambda".
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `scal` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.

The approach is as follows:

1) The training variable `y` (univariate class membership) is transformed to a dummy table (Ydummy) 
    containing nlev columns, where nlev is the number of classes present in `y`. Each column of 
    Ydummy is a dummy (0/1) variable. 
2) Then, a ridge regression (RR) is run on {`X`, Ydummy}, returning predictions of the dummy variables 
    (= object `posterior` returned by fuction `predict`).  These predictions can be considered as unbounded
    estimates (i.e. eventually outside of [0, 1]) of the class membership probabilities.
3) For a given observation, the final prediction is the class corresponding to the dummy variable for which 
    the probability estimate is the highest.

The low-level method (i.e. having argument `weights`) of the function allows to set any vector of observation weights 
to be used in the intermediate computations. In the high-level methods (no argument `weights`), they are automatically 
computed from the argument `prior` value: for each class, the total of the observation weights is set equal to 
the prior probability corresponding to the class.

**Note:** For highly unbalanced classes, it may be recommended to set 'prior = :unif' when using the function
(and to use a score such as `merrp` instead of `errp` when evaluating the perfomance).

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
model = rrda(; lb) 
fit!(model, Xtrain, ytrain)
@names model
@names fitm = model.fitm
fitm.lev
fitm.ni
@names fitm.fitm
aggsumv(fitm.fitm.weights.w, ytrain)

coef(fitm.fitm)

res = predict(model, Xtest) ;
@names res
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest; lb = [.1; .01]).pred
```
""" 
rrda(; kwargs...) = JchemoModel(rrda, nothing, kwargs)

function rrda(X, y; kwargs...)
    par = recovkw(ParRrda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    rrda(X, y, weights; kwargs...)
end

function rrda(X, y, weights::Weight; kwargs...)    
    par = recovkw(ParRrda, kwargs).par
    res = dummy(y)
    ni = tab(y).vals 
    fitm = rr(X, res.Y, weights; kwargs...)
    Rrda(fitm, res.lev, ni, par)
end

"""
    predict(object::Rrda, X; lb = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `lb` : Regularization parameter, or collection of regularization parameters, "lambda" to consider. 
    If nothing, it is the parameter stored in the fitted model.
""" 
function predict(object::Rrda, X; lb = nothing)
    X = ensure_mat(X)
    Q = eltype(X)
    Qy = eltype(object.lev)
    m = nro(X)
    isnothing(lb) ? lb = object.par.lb : nothing
    le_lb = length(lb)
    pred = list(Matrix{Qy}, le_lb)
    posterior = list(Matrix{Q}, le_lb)
    @inbounds for i = 1:le_lb
        zp = predict(object.fitm, X; lb = lb[i]).pred
        z =  mapslices(argmax, zp; dims = 2)  # if equal, argmax takes the first
        pred[i] = reshape(recod_indbylev(z, object.lev), m, 1)
        posterior[i] = zp
    end 
    if le_lb == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end

