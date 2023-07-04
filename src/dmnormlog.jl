struct Dmnormlog
    mu
    Uinv 
    logdetS
    logcst
end

"""
    dmnormlog(X = nothing; mu = nothing, S = nothing,
        simpl::Bool = false)
    dmnormlog!(X = nothing; mu = nothing, S = nothing,
        simpl::Bool = false)
Logarithm of the normal probability density estimation.
* `X` : X-data (n, p) used to estimate the mean and 
    the covariance matrix. If `nothing`, `mu` and `S` 
    must be provided.
* `mu` : Mean vector of the normal distribution. 
    If `nothing`, `mu` is computed by the column-means of `X`.
* `S` : Covariance matrix of the normal distribution.
    If `nothing`, `S` is computed by cov(`X`; corrected = true).
* `simpl` : Boolean. If `true`, the constant term and the determinant 
    in the density formula are set to 1. Default to `false`. 
    See `dmnorm`.

See the help of function `dmnorm`.

## Examples
```julia
using JLD2, CairoMakie

using JchemoData
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "iris.jld2") 
@load db dat
pnames(dat)

X = dat.X[:, 1:4] 
y = dat.X[:, 5]
n = nro(X)
tab(y) 

s = y .== "setosa"
zX = X[s, :]

fm = dmnormlog(zX) ;
pnames(fm)
fm.Uinv 
fm.logdetS
pred = Jchemo.predict(fm, zX).pred
head(pred) 

fm0 = dmnorm(zX) ;
pred0 = Jchemo.predict(fm0, zX).pred
head(log.(pred0))
```
""" 
function dmnormlog(X = nothing; mu = nothing, S = nothing,
        simpl = false)
    isnothing(S) ? zS = nothing : zS = copy(S)
    dmnormlog!(X; mu = mu, S = zS, simpl = simpl)
end

function dmnormlog!(X = nothing; mu = nothing, S = nothing,
        simpl = false)
    !isnothing(X) ? X = ensure_mat(X) : nothing
    if isnothing(mu)
        mu = vec(mean(X, dims = 1))
    end
    if isnothing(S)
        S = cov(X; corrected = true)
    end
    if simpl 
        logcst = 0
        logdetS = 0
    else
        p = nro(S)
        logcst = -p / 2 * log(2 * pi)
        logdetS = logdet(S)
    end  
    U = cholesky!(Hermitian(S)).U
    LinearAlgebra.inv!(U)
    Dmnormlog(mu, U, logdetS, logcst)
end

function predict(object::Dmnormlog, X)
    X = ensure_mat(X)
    mu = reshape(object.mu, 1, length(object.mu))
    d = mahsqchol(X, mu, object.Uinv)
    @. d = object.logcst - object.logdetS / 2 - d / 2
    (pred = d,)
end


    