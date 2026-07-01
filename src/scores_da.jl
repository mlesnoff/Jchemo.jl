## scores: Must return a matrix
## - In particular for errp, create multiple dispatches Y::AbstractVector, Y::AbstractMatrix

"""
    residcla(pred, y) 
Compute the discrimination residual vector (0 = no error, 1 = error).
* `pred` : Predictions.
* `y` : Observed data (class membership).

## Examples
```julia
using Jchemo

Xtrain = rand(10, 5) 
ytrain = rand(["a" ; "b"], 10)
Xtest = rand(4, 5) 
ytest = rand(["a" ; "b"], 4)

model = plsrda(; nlv = 2)
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
residcla(pred, ytest)
```
"""
residcla(pred, Y) = ensure_mat(pred) .!= ensure_mat(Y)

"""
    errp(pred, y)
Compute the classification error rate (ERRP).
* `pred` : Predictions.
* `y` : Observed data (class membership).

## Examples
```julia
using Jchemo

Xtrain = rand(10, 5) 
ytrain = rand(["a" ; "b"], 10)
Xtest = rand(4, 5) 
ytest = rand(["a" ; "b"], 4)

model = plsrda(; nlv = 2)
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
errp(pred, ytest)
```
"""
function errp(pred, y)
    r = residcla(pred, y)
    res = [sum(r)] / nro(y)
    reshape(res, 1, :)
end

"""
    merrp(pred, y)
Compute the mean intra-class classification error rate.
* `pred` : Predictions.
* `y` : Observed data (class membership).

ERRP (see function `errp`) is computed for each class.
Function `merrp` returns the average of these intra-class ERRPs.   

## Examples
```julia
using Jchemo

Xtrain = rand(10, 5) 
ytrain = rand(["a" ; "b"], 10)
Xtest = rand(4, 5) 
ytest = rand(["a" ; "b"], 4)

model = plsrda(; nlv = 2)
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
merrp(pred, ytest)
```
"""
function merrp(pred, y)
    r = residcla(pred, y)
    res = tab(y)
    lev = res.keys
    nlev = length(lev)
    v = zeros(nlev)
    @inbounds for i in eachindex(lev)
        s = y .== res.keys[i]
        v[i] = sum(r[s]) / res.vals[i]
    end
    reshape([mean(v)], 1, :)
end


