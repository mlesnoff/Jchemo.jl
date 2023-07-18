struct Plsrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

"""
    plsrda(X, y, weights = ones(nro(X)); nlv,
        scal::Bool = false)
Discrimination based on partial least squares regression (PLSR-DA).
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights of the observations. Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

This is the usual "PLSDA". 
The training variable `y` (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy (0/1) variable. 
Then, a PLS2 is implemented on `X` and Ydummy, returning `nlv` LVs.
Finally, a multiple linear regression (MLR) is run between the LVs and each 
column of Ydummy, returning predictions of the dummy variables 
(= object `posterior` returned by function `predict`). 
These predictions can be considered as unbounded 
estimates (i.e. eventually outside of [0, 1]) of the class membership probabilities.
For a given observation, the final prediction is the class corresponding 
to the dummy variable for which the probability estimate is the highest.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2") 
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
fm = plsrda(Xtrain, ytrain; nlv = nlv) ;
pnames(fm)
typeof(fm.fm) # = PLS2 model

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.posterior
res.pred
err(res.pred, ytest)
confusion(res.pred, ytest).cnt

Jchemo.transform(fm, Xtest)

Jchemo.transform(fm.fm, Xtest)
Jchemo.coef(fm.fm)
summary(fm.fm, Xtrain)

Jchemo.predict(fm, Xtest; nlv = 1:2).pred
```
""" 
function plsrda(X, y, weights = ones(nro(X)); nlv,
        scal::Bool = false)
    res = dummy(y)
    ni = tab(y).vals
    fm = plskern(X, res.Y, weights; nlv = nlv, scal = scal)
    Plsrda(fm, res.lev, ni)
end

""" 
    transform(object::Plsrda, X; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and a matrix X.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transform(object::Plsrda, X; nlv = nothing)
    transform(object.fm, X; nlv = nlv)
end

"""
    predict(object::Plsrda, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Plsrda, X; nlv = nothing)
    X = ensure_mat(X)
    m = nro(X)
    a = size(object.fm.T, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(le_nlv, Union{Matrix{Int64}, Matrix{Float64}, Matrix{String}})
    posterior = list(le_nlv, Matrix{Float64})
    @inbounds for i = 1:le_nlv
        zpred = predict(object.fm, X; nlv = nlv[i]).pred
        #if softmax
        #    @inbounds for j = 1:m
        #        zpred[j, :] .= mweight(exp.(zpred[j, :]))
        #   end
        #end
        z =  mapslices(argmax, zpred; dims = 2)  # if equal, argmax takes the first
        pred[i] = reshape(replacebylev2(z, object.lev), m, 1)     
        posterior[i] = zpred
    end 
    if le_nlv == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end

