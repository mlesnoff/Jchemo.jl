"""
    dmnormlog(X; kwargs...)
    dmnormlog!(X::Matrix; kwargs...)
    dmnormlog(; kwargs...)
Logarithm of the normal probability density estimation.
    * `X` : X-data (n, p) used to estimate the mean `mu` and 
        the covariance matrix `S`. If `X` is not given, 
        `mu` and `S` must be provided in `kwargs`.
Keyword arguments:
    * `mu` : Mean vector of the normal distribution. 
    * `S` : Covariance matrix of the Normal distribution.
    * `simpl` : Boolean. If `true`, the constant term and 
        the determinant in the Normal density formula are set to 1.

See the help page of function `dmnorm`.

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "iris.jld2") 
@load db dat
pnames(dat)
X = dat.X[:, 1:4] 
y = dat.X[:, 5]
n = nro(X)
tab(y) 

## Example of class Setosa 
s = y .== "setosa"
zX = X[s, :]

mod = model(dmnormlog)
fit!(mod, zX)
fm = mod.fm
pnames(fm)
fm.Uinv 
fm.logdetS
@head pred = predict(mod, zX).pred

## Consistency with dmnorm
mod0 = model(dmnorm)
fit!(mod0, zX)
@head pred0 = predict(mod0, zX).pred
@head log.(pred0)
```
""" 
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

function dmnormlog(; kwargs...)
    par = recovkw(ParDmnorm, kwargs).par
    U = cholesky!(Hermitian(copy(par.S))).U   # cholesky! modifies S
    if par.simpl 
        logcst = 0
        logdetS = 0
    else
        p = nro(S)
        logcst = -p / 2 * log(2 * pi)
        logdetS = logdet(S)
    end
    LinearAlgebra.inv!(U)
    Dmnormlog(par.mu, U, logdetS, logcst, par)
end

function predict(object::Dmnormlog, X)
    X = ensure_mat(X)
    mu = reshape(object.mu, 1, length(object.mu))
    d = mahsqchol(X, mu, object.Uinv)
    @. d = object.logcst - object.logdetS / 2 - d / 2
    (pred = d,)
end


    