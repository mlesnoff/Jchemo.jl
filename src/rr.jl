struct Rr
    V::Array{Float64}
    TtDY::Array{Float64}
    sv::Vector{Float64}
    lb::Float64
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    weights::Vector{Float64}
end

"""
    rr(X, Y, weights = ones(nro(X)); lb = .01,
        scal::Bool = false)
    rr!(X::Matrix, Y::Matrix, weights = ones(nro(X)); lb = .01,
        scal::Bool = false)
Ridge regression (RR) implemented by SVD factorization.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations. Internally normalized to sum to 1. 
* `lb` : Ridge regularization parameter "lambda".
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

`X` and `Y` are internally centered. The model is computed with an intercept. 

## References 
Cule, E., De Iorio, M., 2012. A semi-automatic method to guide the choice 
of ridge parameter in ridge regression. arXiv:1205.0686.

Hastie, T., Tibshirani, R., 2004. Efficient quadratic regularization 
for expression arrays. Biostatistics 5, 329-340. https://doi.org/10.1093/biostatistics/kxh010

Hastie, T., Tibshirani, R., Friedman, J., 2009. The elements of statistical learning: data mining, 
inference, and prediction, 2nd ed. Springer, New York.

Hoerl, A.E., Kennard, R.W., 1970. Ridge Regression: Biased Estimation for Nonorthogonal Problems. 
Technometrics 12, 55-67. https://doi.org/10.1080/00401706.1970.10488634

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

lb = 10^(-2)
fm = rr(Xtrain, ytrain; lb = lb) ;
#fm = rrchol(Xtrain, ytrain; lb = lb) ;
pnames(fm)

zcoef = Jchemo.coef(fm)
zcoef.int
zcoef.B
# Only for rr
Jchemo.coef(fm; lb = .1).B

res = Jchemo.predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
plotxy(vec(res.pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f    

# Only for rr
res = Jchemo.predict(fm, Xtest; lb = [.1 ; .01])
res.pred[1]
res.pred[2]
```
""" 
function rr(X, Y, weights = ones(nro(X)); lb = .01,
    scal::Bool = false)
    rr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; lb = lb, 
        scal = scal)
end

function rr!(X::Matrix, Y::Matrix, weights = ones(nro(X)); lb = .01,
        scal::Bool = false)
    p = nco(X)
    weights = mweight(weights)
    sqrtw = sqrt.(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X, weights)
        cscale!(X, xmeans, xscales)
    else
        center!(X, xmeans)
    end
    center!(Y, ymeans)  
    sqrtD = Diagonal(sqrtw)
    res = LinearAlgebra.svd!(sqrtD * X)
    sv = res.S
    TtDY = Diagonal(sv) * res.U' * (sqrtD * Y)
    Rr(res.V, TtDY, sv, lb, xmeans, xscales, ymeans, weights)
end

"""
    rrchol(X, Y, weights = ones(nro(X)); lb = .01, 
        scal::Bool = false)
    rrchol!(X::Matrix, Y::Matrix, weights = ones(nro(X)); lb = .01,
        scal::Bool = false)
Ridge regression (RR) using the Normal equations and a Cholesky factorization.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations. Internally normalized to sum to 1. 
* `lb` : Ridge regularization parameter "lambda".
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

`X` and `Y` are internally centered. The model is computed with an intercept. 

See `?rr` for eaxamples.

## References 
Cule, E., De Iorio, M., 2012. A semi-automatic method to guide the choice 
of ridge parameter in ridge regression. arXiv:1205.0686.

Hastie, T., Tibshirani, R., 2004. Efficient quadratic regularization 
for expression arrays. Biostatistics 5, 329-340. https://doi.org/10.1093/biostatistics/kxh010

Hastie, T., Tibshirani, R., Friedman, J., 2009. The elements of statistical learning: data mining, 
inference, and prediction, 2nd ed. Springer, New York.

Hoerl, A.E., Kennard, R.W., 1970. Ridge Regression: Biased Estimation for Nonorthogonal Problems. 
Technometrics 12, 55-67. https://doi.org/10.1080/00401706.1970.10488634
""" 
function rrchol(X, Y, weights = ones(nro(X)); lb = .01,
        scal::Bool = false)
    rrchol!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; lb = lb,
        scal = scal)
end

function rrchol!(X::Matrix, Y::Matrix, weights = ones(nro(X)); lb = .01,
        scal::Bool = false)
    @assert nco(X) > 1 "Method only working for X with > 1 column."
    p = nco(X)
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X, weights)
        cscale!(X, xmeans, xscales)
    else
        center!(X, xmeans)
    end
    center!(Y, ymeans)  
    XtD = X' * Diagonal(weights)
    B = cholesky!(Hermitian(XtD * X + lb^2 * Diagonal(ones(p)))) \ (XtD * Y)
    B .= Diagonal(1 ./ xscales) * B 
    int = ymeans' .- xmeans' * B
    Mlr(B, int, weights)
end

"""
    coef(object::Rr; lb = nothing)
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
* `lb` : Ridge regularization parameter "lambda".
    If nothing, it is the parameter stored in the fitted model.
""" 
function coef(object::Rr; lb = nothing)
    isnothing(lb) ? lb = object.lb : nothing
    eig = object.sv.^2
    z = 1 ./ (eig .+ lb^2)
    beta = Diagonal(z) * object.TtDY
    B = Diagonal(1 ./ object.xscales) * object.V * beta
    int = object.ymeans' .- object.xmeans' * B
    tr = sum(eig .* z)
    (B = B, int = int, df = 1 + tr)
end

"""
    predict(object::Rr, X; lb = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `lb` : Regularization parameter, or collection of regularization parameters, 
    "lambda" to consider. If nothing, it is the parameter stored in the 
    fitted model.
""" 
function predict(object::Rr, X; lb = nothing)
    X = ensure_mat(X)
    isnothing(lb) ? lb = object.lb : nothing
    le_lb = length(lb)
    pred = list(le_lb, Matrix{Float64})
    @inbounds for i = 1:le_lb
        z = coef(object; lb = lb[i])
        pred[i] = z.int .+ X * z.B
    end 
    le_lb == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end
