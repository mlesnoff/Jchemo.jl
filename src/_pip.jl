
struct Pipeline
    model::Tuple
end

"""
    pip(args...)
Build a pipeline of models.
* `args...` : Succesive models, see examples.

## Examples
```julia
using JLD2, CairoMakie, JchemoData
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
@names dat
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)
ntrain = nro(Xtrain)
ntest = nro(Xtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)

## Pipeline Snv :> Savgol :> Pls :> Svmr

model1 = snv()
model2 = savgol(npoint = 11, deriv = 2, degree = 3)
model3 = plskern(nlv = 15)
model4 = svmr(gamma = 1e3, cost = 1000, epsilon = .1)
model = pip(model1, model2, model3, model4)
fit!(model, Xtrain, ytrain)
res = predict(model, Xtest) ; 
@head res.pred 
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",
      ylabel = "Observed").f
```
"""
pip(args...) = Pipeline(values(args))

function fit!(model::Pipeline, X, Y = nothing; verbose = :false)
    K = length(model.model)
    @assert K > 1 "A pipeline must contain at least 2 models."
    typ = list(Symbol, K)
    for i = 1:K
        meth = methods(model.model[i].algo)
        res = Base.method_argnames.(meth)[2]  # assume here that the working methods of the function start from the 2nd
        z = uppercase.(String.(res))
        (in("X", z) || in("XBL", z)) && in("Y", z) ? typ[i] = :XY : typ[i] = :X
    end
    verbose ? show(typ) : nothing
    @inbounds for i = 1:(K - 1)
        if typ[i] == :X
            fit!(model.model[i], X)
        elseif typ[i] == :XY
            fit!(model.model[i], X, Y)
        end 
        X = transf(model.model[i], X)
    end
    if typ[K] == :X
        fit!(model.model[K], X)
    elseif typ[K] == :XY
        fit!(model.model[K], X, Y)
    end 
end

function transf(model::Pipeline, X)  
    K = length(model.model)
    @assert K > 1 "Wrong pipeline: must contain at least 2 models."
    @inbounds for i = 1:K
        X = transf(model.model[i], X)
    end
    X
end

function predict(model::Pipeline, X)
    K = length(model.model)
    @assert K > 1 "Wrong pipeline: must contain at least 2 models."
    @inbounds for i = 1:(K - 1)
        X = transf(model.model[i], X)
    end
    predict(model.model[K], X)
end


