"""
    bias(pred, Y)
Compute the prediction bias, i.e. the opposite 
    of the mean prediction error.
* `pred` : Predictions.
* `Y` : Observed data.

## Examples
```julia
Xtrain = rand(10, 5) 
Ytrain = rand(10, 2)
ytrain = Ytrain[:, 1]
Xtest = rand(4, 5) 
Ytest = rand(4, 2)
ytest = Ytest[:, 1]

mod = plskern(nlv = 2)
fit!(mod, Xtrain, Ytrain)
pred = predict(mod, Xtest).pred
bias(pred, Ytest)

mod = plskern(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
bias(pred, ytest)
```
"""
function bias(pred, Y)
    r = residreg(pred, Y)
    reshape(-colmean(r), 1, :)
end

"""
    cor2(pred, Y)
Compute the squared linear correlation between 
    data and predictions.
* `pred` : Predictions.
* `Y` : Observed data.

## Examples
```julia
Xtrain = rand(10, 5) 
Ytrain = rand(10, 2)
ytrain = Ytrain[:, 1]
Xtest = rand(4, 5) 
Ytest = rand(4, 2)
ytest = Ytest[:, 1]

mod = plskern(nlv = 2)
fit!(mod, Xtrain, Ytrain)
pred = predict(mod, Xtest).pred
cor2(pred, Ytest)

mod = plskern(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
cor2(pred, ytest)
```
"""
function cor2(pred, Y)
    Y = ensure_mat(Y)
    q = nco(Y)
    res = cor(pred, Y).^2
    q == 1 ? res = [res; ] : res = diag(res)
    reshape(res, 1, :)
end

"""
    errp(pred, y)
Compute the classification error rate (ERR).
* `pred` : Predictions.
* `y` : Observed data.

## Examples
```julia
Xtrain = rand(10, 5) 
ytrain = rand(["a" ; "b"], 10)
Xtest = rand(4, 5) 
ytest = rand(["a" ; "b"], 4)

mod = plsrda(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
errp(pred, ytest)
```
"""
function errp(pred, y)
    r = residcla(pred, y)
    res = [sum(r)] / nro(y)
    reshape(res, 1, :)
end

"""
    mse(pred, Y; digits = 3)
Summary of model performance for regression.
* `pred` : Predictions.
* `Y` : Observed data.

## Examples
```julia
Xtrain = rand(10, 5) 
Ytrain = rand(10, 2)
ytrain = Ytrain[:, 1]
Xtest = rand(4, 5) 
Ytest = rand(4, 2)
ytest = Ytest[:, 1]

mod = plskern(nlv = 2)
fit!(mod, Xtrain, Ytrain)
pred = predict(mod, Xtest).pred
mse(pred, Ytest)

mod = plskern(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
mse(pred, ytest)
```
"""
function mse(pred, Y; digits = 3)
    q = nco(Y)
    zmsep = msep(pred, Y)
    zrmsep = sqrt.(zmsep)
    zsep = sep(pred, Y)
    zbias = bias(pred, Y)
    zcor2 = cor2(pred, Y)    
    zr2 = r2(pred, Y)
    zrpd = rpd(pred, Y)
    zrpdr = rpdr(pred, Y)
    zmean = reshape(colmean(Y), 1, :)
    nam = map(string, repeat(["y"], q), 1:q)
    nam = reshape(nam, 1, :)
    res = (nam = nam, msep = zmsep, rmsep = zrmsep, sep = zsep, bias = zbias, 
        cor2 = zcor2, r2 = zr2, rpd = zrpd, rpdr = zrpdr, mean = zmean)
    res = map(vec, res)
    res = DataFrame(res)  
    res[:, 2:end] = round.(res[:, 2:end], digits = digits)
    res
end

"""
    msep(pred, Y)
Compute the mean of the squared prediction 
    errors (MSEP).
* `pred` : Predictions.
* `Y` : Observed data.

## Examples
```julia
Xtrain = rand(10, 5) 
Ytrain = rand(10, 2)
ytrain = Ytrain[:, 1]
Xtest = rand(4, 5) 
Ytest = rand(4, 2)
ytest = Ytest[:, 1]

mod = plskern(nlv = 2)
fit!(mod, Xtrain, Ytrain)
pred = predict(mod, Xtest).pred
msep(pred, Ytest)

mod = plskern(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
msep(pred, ytest)
```
"""
function msep(pred, Y)
    r = residreg(pred, Y)
    reshape(colmean(r.^2), 1, :)
end

"""
    r2(pred, Y)
Compute the R2 coefficient.
* `pred` : Predictions.
* `Y` : Observed data.

The rate R2 is calculated by:
* R2 = 1 - MSEP(current model) / MSEP(null model) 
where the "null model" is the overall mean. 
For predictions over CV or test sets, and/or for 
non linear models, it can be different from the square 
of the correlation coefficient (`cor2`) between the true 
data and the predictions. 

## Examples
```julia
Xtrain = rand(10, 5) 
Ytrain = rand(10, 2)
ytrain = Ytrain[:, 1]
Xtest = rand(4, 5) 
Ytest = rand(4, 2)
ytest = Ytest[:, 1]

mod = plskern(nlv = 2)
fit!(mod, Xtrain, Ytrain)
pred = predict(mod, Xtest).pred
r2(pred, Ytest)

mod = plskern(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
r2(pred, ytest)
```
"""
function r2(pred, Y)
    m = nro(Y)
    ymeans = colmean(Y)
    M = reduce(hcat, fill(ymeans, m, 1))'
    1 .- msep(pred, Y) ./ msep(M, Y)
end

"""
    residcla(pred, y) 
Compute discrimination residual error 
    (0 = no error, 1 = error).
* `pred` : Predictions.
* `y` : Observed data.

## Examples
```julia
Xtrain = rand(10, 5) 
ytrain = rand(["a" ; "b"], 10)
Xtest = rand(4, 5) 
ytest = rand(["a" ; "b"], 4)

mod = plsrda(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
residcla(pred, ytest)
```
"""
residcla(pred, Y) = pred .!= ensure_mat(Y)

"""
    residreg(pred, Y) 
Compute regression prediction errors.
* `pred` : Predictions.
* `Y` : Observed data.

## Examples
```julia
Xtrain = rand(10, 5) 
Ytrain = rand(10, 2)
ytrain = Ytrain[:, 1]
Xtest = rand(4, 5) 
Ytest = rand(4, 2)
ytest = Ytest[:, 1]

mod = plskern(nlv = 2)
fit!(mod, Xtrain, Ytrain)
pred = predict(mod, Xtest).pred
residreg(pred, Ytest)

mod = plskern(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
residreg(pred, ytest)
```
"""
residreg(pred, Y) = ensure_mat(Y) - pred

"""
    rmsep(pred, Y)
Compute the square root of the mean of the squared 
    prediction errors (RMSEP).
* `pred` : Predictions.
* `Y` : Observed data.

## Examples
```julia
Xtrain = rand(10, 5) 
Ytrain = rand(10, 2)
ytrain = Ytrain[:, 1]
Xtest = rand(4, 5) 
Ytest = rand(4, 2)
ytest = Ytest[:, 1]

mod = plskern(nlv = 2)
fit!(mod, Xtrain, Ytrain)
pred = predict(mod, Xtest).pred
rmsep(pred, Ytest)

mod = plskern(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
rmsep(pred, ytest)
```
"""
rmsep(pred, Y) = sqrt.(msep(pred, Y))

"""
    rmsepstand(pred, Y)
Compute the standardized square root of the 
    mean of the squared prediction errors 
    (RMSEP_stand).
* `pred` : Predictions.
* `Y` : Observed data.

RMSEP is standardized to `Y`: 
* RMSEP_stand = RMSEP ./ `Y`.

## Examples
```julia
Xtrain = rand(10, 5) 
Ytrain = rand(10, 2)
ytrain = Ytrain[:, 1]
Xtest = rand(4, 5) 
Ytest = rand(4, 2)
ytest = Ytest[:, 1]

mod = plskern(nlv = 2)
fit!(mod, Xtrain, Ytrain)
pred = predict(mod, Xtest).pred
rmsepstand(pred, Ytest)

mod = plskern(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
rmsepstand(pred, ytest)
```
"""
function rmsepstand(pred, Y)
    Y = ensure_mat(Y)
    rmsep(pred ./ Y, ones(size(Y)))
end

"""
    rpd(pred, Y)
Compute the ratio "deviation to model performance" 
    (RPD).
* `pred` : Predictions.
* `Y` : Observed data.

This is the ratio of the deviation to the model performance 
to the deviation, defined by:
* RPD = Std(Y) / RMSEP
where Std(Y) is the standard deviation. 

Since Std(Y) = RMSEP(null model) where the null model is 
the simple average, this also gives:
* RPD = RMSEP(null model) / RMSEP 

## Examples
```julia
Xtrain = rand(10, 5) 
Ytrain = rand(10, 2)
ytrain = Ytrain[:, 1]
Xtest = rand(4, 5) 
Ytest = rand(4, 2)
ytest = Ytest[:, 1]

mod = plskern(nlv = 2)
fit!(mod, Xtrain, Ytrain)
pred = predict(mod, Xtest).pred
rpd(pred, Ytest)

mod = plskern(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
rpd(pred, ytest)
```
"""
function rpd(pred, Y)
    Y = ensure_mat(Y)
    std(Y; dims = 1, corrected = false) ./ rmsep(pred, Y)
end 

"""
    rpdr(pred, Y)
Compute a robustified RPD.
* `pred` : Predictions.
* `Y` : Observed data.

## Examples
```julia
Xtrain = rand(10, 5) 
Ytrain = rand(10, 2)
ytrain = Ytrain[:, 1]
Xtest = rand(4, 5) 
Ytest = rand(4, 2)
ytest = Ytest[:, 1]

mod = plskern(nlv = 2)
fit!(mod, Xtrain, Ytrain)
pred = predict(mod, Xtest).pred
rpdr(pred, Ytest)

mod = plskern(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
rpdr(pred, ytest)
```
"""
function rpdr(pred, Y)
    Y = ensure_mat(Y)
    u = mapslices(Jchemo.mad, Y; dims = 1) / 1.4826
    r = residreg(pred, Y)
    v = mapslices(median, abs.(r); dims = 1)
    res = u ./ v
    reshape(res, 1, :)
end

"""
    sep(pred, Y)
Compute the corrected SEP ("SEP_c"), i.e. the standard 
    deviation of the prediction errors.
* `pred` : Predictions.
* `Y` : Observed data.

## References
Bellon-Maurel, V., Fernandez-Ahumada, E., Palagos, B., 
Roger, J.-M., McBratney, A., 2010. Critical review of 
chemometric indicators commonly used for assessing the 
quality of the prediction of soil attributes by NIR 
spectroscopy. TrAC Trends in Analytical Chemistry 29, 
1073–1081. 
https://doi.org/10.1016/j.trac.2010.05.006

## Examples
```julia
Xtrain = rand(10, 5) 
Ytrain = rand(10, 2)
ytrain = Ytrain[:, 1]
Xtest = rand(4, 5) 
Ytest = rand(4, 2)
ytest = Ytest[:, 1]

mod = plskern(nlv = 2)
fit!(mod, Xtrain, Ytrain)
pred = predict(mod, Xtest).pred
sep(pred, Ytest)

mod = plskern(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
sep(pred, ytest)
```
"""
sep(pred, Y) = sqrt.(msep(pred, Y) .- bias(pred, Y).^2)

"""
    ssr(pred, Y)
Compute the sum of squared prediction errors (SSR).
* `pred` : Predictions.
* `Y` : Observed data.

## Examples
```julia
Xtrain = rand(10, 5) 
Ytrain = rand(10, 2)
ytrain = Ytrain[:, 1]
Xtest = rand(4, 5) 
Ytest = rand(4, 2)
ytest = Ytest[:, 1]

mod = plskern(nlv = 2)
fit!(mod, Xtrain, Ytrain)
pred = predict(mod, Xtest).pred
ssr(pred, Ytest)

mod = plskern(nlv = 2)
fit!(mod, Xtrain, ytrain)
pred = predict(mod, Xtest).pred
ssr(pred, ytest)
```
"""
function ssr(pred, Y)
    r = residreg(pred, Y)
    reshape(sum(r.^2, dims = 1), 1, :)
end

## scores:
## - Must return a matrix
## - In particular for errp, create multiple 
## dispatches Y::AbstractVector, Y::AbstractMatrix
