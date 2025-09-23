"""
    dmnormlog(; kwargs...)
    dmnormlog(X; kwargs...)
    dmnormlog!(X::Matrix; kwargs...)
    dmnormlog(mu, S; kwargs...)
    dmnormlog!(mu::Vector, S::Matrix; kwargs...)
Logarithm of the normal probability density estimation.
    * `X` : X-data (n, p) used to estimate the mean `mu` and the covariance matrix `S`. If `X` is not given, 
        `mu` and `S` must be provided in `kwargs`.
    * `mu` : Mean vector of the normal distribution. 
    * `S` : Covariance matrix of the Normal distribution.
Keyword arguments:
    * `simpl` : Boolean. If `true`, the constant term and the determinant in the Normal density formula 
        are set to 1.

See the help page of function `dmnorm`.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "iris.jld2") 
@load db dat
@names dat
X = dat.X[:, 1:4] 
y = dat.X[:, 5]
n = nro(X)
tab(y) 

## Example of class Setosa 
s = y .== "setosa"
zX = X[s, :]

model = dmnormlog()
fit!(model, zX)
fitm = model.fitm
@names fitm
fitm.Uinv 
fitm.logdetS
@head pred = predict(model, zX).pred

## Consistency with dmnorm
model0 = dmnorm()
fit!(model0, zX)
@head pred0 = predict(model0, zX).pred
@head log.(pred0)
```
""" 
dmnormlog(; kwargs...) = JchemoModel(dmnormlog, nothing, kwargs)

function dmnormlog(X; kwargs...)
    dmnormlog!(copy(ensure_mat(X)); kwargs...)
end

function dmnormlog!(X::Matrix; kwargs...)
    par = recovkw(ParDmnorm, kwargs).par
    mu = colmean(X) 
    S = cov(X; corrected = true)
    if par.simpl 
        logcst = 0
        logdetS = 0
    else
        p = nro(S)
        logcst = -p / 2 * log(2 * pi)
        logdetS = logdet(S)
    end
    U = cholesky!(Hermitian(S)).U    # cholesky! modifies S
    LinearAlgebra.inv!(U)
    Dmnormlog(mu, U, logdetS, logcst, par)
end

function dmnormlog(mu, S; kwargs...)
    dmnormlog!(copy(vec(mu)), copy(ensure_mat(S)); kwargs...) 
end

function dmnormlog!(mu::Vector, S::Matrix; kwargs...)
    par = recovkw(ParDmnorm, kwargs).par
    U = cholesky!(Hermitian(copy(S))).U   # cholesky! modifies S
    if par.simpl 
        logcst = 0
        logdetS = 0
    else
        p = nro(S)
        logcst = -p / 2 * log(2 * pi)
        logdetS = logdet(S)
    end
    LinearAlgebra.inv!(U)
    Dmnormlog(mu, U, logdetS, logcst, par)
end

function predict(object::Dmnormlog, X)
    X = ensure_mat(X)
    mu = reshape(object.mu, 1, length(object.mu))
    d = mahsqchol(X, mu, object.Uinv)
    @. d = object.logcst - object.logdetS / 2 - d / 2
    (pred = d,)
end


    