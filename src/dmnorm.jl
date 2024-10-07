"""
    dmnorm(X; kwargs...)
    dmnorm!(X::Matrix; kwargs...)
    dmnorm(; kwargs...)
Normal probability density estimation.
* `X` : X-data (n, p) used to estimate the mean `mu` and 
    the covariance matrix `S`. If `X` is not given, 
    `mu` and `S` must be provided in `kwargs`.
Keyword arguments:
* `mu` : Mean vector of the normal distribution. 
* `S` : Covariance matrix of the Normal distribution.
* `simpl` : Boolean. If `true`, the constant term and 
    the determinant in the Normal density formula are set to 1.

Data `X` can be univariate (p = 1) or multivariate (p > 1). See examples.

When `simple` = `true`, the determinant of the covariance matrix 
(object `detS`) and the constant (2 * pi)^(-p / 2) (object `cst`) 
in the density formula are set to 1. The function returns a pseudo 
density that resumes to exp(-d / 2), where d is the squared Mahalanobis 
distance to the center `mu`. This can for instance be useful when the number 
of columns (p) of `X` becomes too large, with the possible consequences
that:
* `detS` tends to 0 or, conversely, to infinity;
* `cst` tends to 0,
which makes impossible to compute the true density. 

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

nlv = 2
mod0 = fda; nlv)
fit!(mod0, X, y)
@head T = transf(mod0, X)
n, p = size(T)

#### Probability density in the FDA score space (2D)
#### Example of class Setosa 
s = y .== "setosa"
zT = T[s, :]
m = nro(zT)

#### Bivariate distribution
model = dmnorm)
fit!(model, zT)
fitm = model.fitm
pnames(fitm)
fitm.Uinv 
fitm.detS
pred = predict(model, zT).pred
@head pred

mu = colmean(zT)
S = covm(zT, mweight(ones(m))) * m / (m - 1) # corrected cov. matrix
## Direct syntax
fitm = dmnorm(; mu, S) ; 
pnames(fitm)
fitm.Uinv
fitm.detS

npoints = 2^7
lims = [(minimum(zT[:, j]), maximum(zT[:, j])) for j = 1:nlv]
x1 = LinRange(lims[1][1], lims[1][2], npoints)
x2 = LinRange(lims[2][1], lims[2][2], npoints)
z = mpar(x1 = x1, x2 = x2)
grid = reduce(hcat, z)
model = dmnorm)
fit!(model, zT)
res = predict(model, grid) ;
pred_grid = vec(res.pred)
f = Figure(size = (600, 400))
ax = Axis(f[1, 1];  title = "Density for FDA scores (Iris - Setosa)", 
    xlabel = "Score 1", ylabel = "Score 2")
co = contour!(ax, grid[:, 1], grid[:, 2], pred_grid; levels = 10, labels = true)
scatter!(ax, T[:, 1], T[:, 2], color = :red, markersize = 5)
scatter!(ax, zT[:, 1], zT[:, 2], color = :blue, markersize = 5)
#xlims!(ax, -12, 12) ;ylims!(ax, -12, 12)
f

#### Univariate distribution
j = 1
x = zT[:, j]
model = dmnorm)
fit!(model, x)
pred = predict(model, x).pred 
f = Figure()
ax = Axis(f[1, 1]; xlabel = string("FDA-score ", j))
hist!(ax, x; bins = 30, normalization = :pdf)  # area = 1
scatter!(ax, x, vec(pred); color = :red)
f

x = zT[:, j]
npoints = 2^8
lims = [minimum(x), maximum(x)]
#delta = 5 ; lims = [minimum(x) - delta, maximum(x) + delta]
grid = LinRange(lims[1], lims[2], npoints)
model = dmnorm)
fit!(model, x)
pred_grid = predict(model, grid).pred 
f = Figure()
ax = Axis(f[1, 1]; xlabel = string("FDA-score ", j))
hist!(ax, x; bins = 30, normalization = :pdf)  # area = 1
lines!(ax, grid, vec(pred_grid); color = :red)
f
```
"""
function dmnorm(X; kwargs...)
    dmnorm!(copy(ensure_mat(X)); kwargs...)
end

function dmnorm!(X::Matrix; kwargs...)
    par = recovkw(ParDmnorm, kwargs).par
    mu = colmean(X) 
    S = cov(X; corrected = true)
    U = cholesky!(Hermitian(S)).U    # cholesky! modifies S
    if par.simpl 
        cst = 1
        detS = 1
    else
        p = nro(S)
        cst = (2 * pi)^(-p / 2)
        detS = det(U)^2  
    end
    LinearAlgebra.inv!(U)
    #cholesky!(S)
    #U = sqrt(diag(diag(S), nrow = p))
    #Uinv = solve(diag(diag(S), nrow = p))
    Dmnorm(mu, U, detS, cst, par)
end

function dmnorm(; kwargs...)
    par = recovkw(ParDmnorm, kwargs).par
    U = cholesky!(Hermitian(copy(par.S))).U   # cholesky! modifies S
    if par.simpl 
        cst = 1
        detS = 1
    else
        p = nro(par.S)
        cst = (2 * pi)^(-p / 2)
        detS = det(U)^2  
    end
    LinearAlgebra.inv!(U)
    Dmnorm(par.mu, U, detS, cst, par)
end

"""
    predict(object::Dmnorm, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : Data (vector) for which predictions are computed.
""" 
function predict(object::Dmnorm, X)
    X = ensure_mat(X)
    mu = reshape(object.mu, 1, length(object.mu))
    d = mahsqchol(X, mu, object.Uinv)
    @. d = object.cst / sqrt(object.detS) * exp(-d / 2)  # = density
    (pred = d,)
end



        
    