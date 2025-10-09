"""
    rr(; kwargs...)
    rr(X, Y; kwargs...)
    rr(X, Y, weights::Weight; kwargs...)
    rr!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
Ridge regression (RR) implemented by SVD factorization.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `lb` : Ridge regularization parameter 'lambda'.
* `scal` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.

The function computes a model with intercept. After `X` and y (a given column of `Y`) have been
centered (an `X` eventually scaled) and weighted by sqrtw = sqrt.(`weights.w`), the function finds 
b (q, 1) (the corresponding column of output `B` (p, q)) that minimizes 
* ||y - X * b||^2 + `lb`^2 * ||b||^2 
where ||.|| is the Euclidean norm.

## References 
Cule, E., De Iorio, M., 2012. A semi-automatic method to guide the choice of ridge parameter 
in ridge regression. arXiv:1205.0686.

Hastie, T., Tibshirani, R., 2004. Efficient quadratic regularization for expression arrays. 
Biostatistics 5, 329-340. https://doi.org/10.1093/biostatistics/kxh010

Hastie, T., Tibshirani, R., Friedman, J., 2009. The elements of statistical learning: data mining, 
inference, and prediction, 2nd ed. Springer, New York.

Hoerl, A.E., Kennard, R.W., 1970. Ridge Regression: Biased Estimation for Nonorthogonal Problems. 
Technometrics 12, 55-67. https://doi.org/10.1080/00401706.1970.10488634

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
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

lb = 1e-3
model = rr(; lb) 
#model = rrchol(; lb) 
fit!(model, Xtrain, ytrain)
@names model
@names model.fitm

coef(model)

res = predict(model, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

## !! Only for function 'rr' (not for 'rrchol')
coef(model; lb = 1e-1)
res = predict(model, Xtest; lb = [.1 ; .01])
@head res.pred[1]
@head res.pred[2]
```
""" 
rr(; kwargs...) = JchemoModel(rr, nothing, kwargs)

function rr(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    rr(X, Y, weights; kwargs...)
end

function rr(X, Y, weights::Weight; kwargs...)
    rr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function rr!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
    par = recovkw(ParRr, kwargs).par
    Q = eltype(X)
    isa(Y, BitMatrix) ? Y = convert.(Q, Y) : nothing
    p = nco(X)
    sqrtw = sqrt.(weights.w)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    fcenter!(Y, ymeans)
    rweight!(X, sqrtw)
    rweight!(Y, sqrtw)
    res = LinearAlgebra.svd!(X)
    sv = res.S
    TtY = Diagonal(sv) * res.U' * Y
    Rr(res.V, TtY, sv, xmeans, xscales, ymeans, weights, par)
end

"""
    coef(object::Rr; lb = nothing)
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
* `lb` : Ridge regularization parameter 'lambda'.
""" 
function coef(object::Rr; lb = nothing)
    isnothing(lb) ? lb = object.par.lb : nothing
    lb = convert(eltype(object.sv), lb)
    eig = object.sv.^2
    z = 1 ./ (eig .+ lb^2)
    beta = Diagonal(z) * object.TtY
    B = rweight(object.V, 1 ./ object.xscales) * beta
    int = object.ymeans' .- object.xmeans' * B
    tr = sum(eig .* z)
    (B = B, int = int, df = 1 + tr)
end

"""
    predict(object::Rr, X; lb = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `lb` : Regularization parameter, or collection of regularization parameters, 'lambda' to consider.
""" 
function predict(object::Rr, X; lb = nothing)
    X = ensure_mat(X)
    Q = eltype(X)
    isnothing(lb) ? lb = object.par.lb : nothing
    le_lb = length(lb)
    pred = list(Matrix{Q}, le_lb)
    @inbounds for i = 1:le_lb
        z = coef(object; lb = lb[i])
        pred[i] = z.int .+ X * z.B
    end 
    le_lb == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end
