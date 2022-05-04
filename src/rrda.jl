struct Rrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

struct KrrDa
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

"""
    rrda(X, y, weights = ones(size(X, 1)); lb)
Discrimination based on ridge regression (RR-DA).
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights of the observations.
* `lb` : A value of the regularization parameter "lambda".

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
using JLD2, CairoMakie
mypath = joinpath(@__DIR__, "..", "data")
db = string(mypath, "\\", "forages.jld2") 
@load db dat
pnames(dat)

Xtrain = dat.Xtrain
ytrain = dat.Ytrain.y
Xtest = dat.Xtest
ytest = dat.Ytest.y

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
function rrda(X, y, weights = ones(size(X, 1)); lb)
    z = dummy(y)
    fm = rr(X, z.Y, weights; lb = lb)
    Rrda(fm, z.lev, z.ni)
end

"""
    krrda(X, y, weights = ones(size(X, 1)); lb, kern = "krbf", kwargs...)
Discrimination based on kernel ridge regression (KRR-DA).
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations.
* `lb` : A value of the regularization parameter "lambda".
* Other arguments: see '?kplsr'.

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
using JLD2, CairoMakie
mypath = joinpath(@__DIR__, "..", "data")
db = string(mypath, "\\", "forages.jld2") 
@load db dat
pnames(dat)

Xtrain = dat.Xtrain
ytrain = dat.Ytrain.y
Xtest = dat.Xtest
ytest = dat.Ytest.y

tab(ytrain)
tab(ytest)

gamma = .01
lb = .001
fm = krrda(Xtrain, ytrain; lb = lb, gamma = gamma) ;    
pnames(fm)
pnames(fm.fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.pred
err(res.pred, ytest)

Jchemo.predict(fm, Xtest; lb = [.1; .01]).pred
```
""" 
function krrda(X, y, weights = ones(size(X, 1)); lb, kern = "krbf", kwargs...)
    z = dummy(y)
    fm = krr(X, z.Y, weights; lb = lb, kern = kern, kwargs...)
    KrrDa(fm, z.lev, z.ni)
end

"""
    predict(object::Union{Rrda, KrrDa}, X; lb = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `lb` : Regularization parameter, or collection of regularization parameters, 
    "lambda" to consider. If nothing, it is the parameter stored in the 
    fitted model.
""" 
function predict(object::Union{Rrda, KrrDa}, X; lb = nothing)
    X = ensure_mat(X)
    m = size(X, 1)
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



