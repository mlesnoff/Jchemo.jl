"""
    rrda(X, y, weights = ones(nro(X)); lb)
Discrimination based on ridge regression (RR-DA).
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights of the observations. Internally normalized to sum to 1. 
* `lb` : Ridge regularization parameter "lambda".
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

The training variable `y` (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy (0/1) variable. 
Then, a RR is implemented on the `y` and each column of Ydummy,
returning predictions of the dummy variables (= object `posterior` returned by 
function `predict`). 
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

lb = .001
fm = rrda(Xtrain, ytrain; lb = lb) ;    
pnames(fm)
pnames(fm.fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.pred
err(res.pred, ytest)

Jchemo.predict(fm, Xtest; lb = [.1; .01]).pred
```
""" 
function rrda(X, y, weights = ones(nro(X)); lb,
        scal::Bool = false)
    res = dummy(y)
    ni = tab(y).vals 
    fm = rr(X, res.Y, weights; lb = lb, 
        scal = scal)
    Rrda(fm, res.lev, ni)
end

"""
    predict(object::Rrda, X; lb = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `lb` : Regularization parameter, or collection of regularization parameters, 
    "lambda" to consider. If nothing, it is the parameter stored in the 
    fitted model.
""" 
function predict(object::Rrda, X; lb = nothing)
    X = ensure_mat(X)
    m = nro(X)
    isnothing(lb) ? lb = object.fm.lb : nothing
    le_lb = length(lb)
    pred = list(le_lb, Union{Matrix{Int64}, Matrix{Float64}, Matrix{String}})
    posterior = list(le_lb, Matrix{Float64})
    @inbounds for i = 1:le_lb
        zp = predict(object.fm, X; lb = lb[i]).pred
        z =  mapslices(argmax, zp; dims = 2)  # if equal, argmax takes the first
        pred[i] = reshape(replacebylev2(z, object.lev), m, 1)
        posterior[i] = zp
    end 
    if le_lb == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end

