"""
    rrda(; kwargs...)
    rrda(X, y; kwargs...)
    rrda(X::Matrix{Q}, y::Vector{String}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float
Discrimination based on ridge regression (RR-DA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n). Must be a `Vector{String}`.
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`). 
Keyword arguments: 
* `lb` : Ridge regularization parameter "lambda".
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `scal` : Symbol defining the column scaling of `X`. Possible values are: `:none`, `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

The approach is as follows:

1) The training variable `y` (univariate class membership) is transformed to a dummy table (Ydummy) 
    containing nlev columns, where nlev is the number of classes present in `y`. Each column of 
    Ydummy is a dummy (0/1) variable. 
2) Then, a ridge regression (RR) is run on the data {`X`, Ydummy}, returning predictions of the dummy variables 
    (= object `posterior` returned by fuction `predict`).  These predictions can be considered as unbounded
    estimates (i.e. eventually outside of [0, 1]) of the class membership probabilities.
3) For a given observation, the final prediction is the class corresponding to the dummy variable for which 
    the probability estimate is the highest.

The low-level function method (i.e. having argument `weights`) requires to set as input a vector of observation 
weights. In that case, argument `prior` has no effect: the class prior probabilities (output `priors`) are always 
computed by summing the observation weights by class.

In the high-level methods (no argument `weights`), argument `prior` defines how are preliminary computed the 
observation weights (see function `pweightcla`) that are then given as input in the hidden low level method.

**Note:** For highly unbalanced classes, it may be recommended to define equal class weights ('prior = :unif'),
and to use a performance score such as `merrp`, instead of `errp`.

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
fitm = model.fitm ;
typeof(fitm)
@names fitm
typeof(fitm.fitm_emb) 
@names fitm.fitm_emb

fitm.lev
fitm.ni
fitm.priors

coef(fitm.fitm_emb)

res = predict(model, Xtest) ;
@names res
@head res.posterior
@head res.pred
errp(res.pred, ytest)
conf(res.pred, ytest).cnt

predict(model, Xtest, [.1; .01]).pred
```
""" 
rrda(; kwargs...) = JchemoModel(rrda, nothing, kwargs)

function rrda(X, y; kwargs...)
    par = recovkw(ParRrda{Q}, kwargs).par
    Q = eltype(X[1, 1])
    weights = pweightcla(Q, y; prior = par.prior)
    rrda(X, y, weights; kwargs...)
end

function rrda(X::Matrix{Q}, y::Vector{String}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: Float    
    par = recovkw(ParRrda{Q}, kwargs).par
    res = dummy(Q, y)
    ni = tab(y).vals 
    priors = aggsumv(weights.values, vec(y)).val  # output not used, only for information
    fitm_emb = rr(X, res.Y, weights; kwargs...)
    Rrda(fitm_emb, ni, priors, res.lev, par)
end

"""
    predict(object::Rrda, X)
    predict(object::Rrda, X, lb::Union{T, AbstractVector{T}})  where T <: Float
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `lb` : Regularization parameter, or collection of regularization parameters, "lambda" to consider. 
""" 
function predict(object::Rrda, X)
    X = ensure_mat(X)
    m = nro(X)
    res = predict(object.fitm_emb, X)
    v =  mapslices(argmax, res.pred; dims = 2)  # if equal, argmax takes the first
    pred = reshape(recod_indbylev(v, object.lev), m, 1)
    (pred = pred, posterior = res.pred, lb = res.lb)
end

function predict(object::Rrda, X, lb::Union{T, AbstractVector{T}})  where T <: Float
    X = ensure_mat(X)
    m = nro(X)
    Qy = eltype(object.lev)
    res = predict(object.fitm_emb, X, lb)
    le_lb = length(lb) 
    pred = list(Matrix{Qy}, le_lb)
    @inbounds for i in eachindex(lb)
        v =  mapslices(argmax, res.pred[i]; dims = 2)  # if equal, argmax takes the first
        pred[i] = reshape(recod_indbylev(v, object.lev), m, 1)
    end 
    (pred = pred, posterior = res.pred, lb)
end

