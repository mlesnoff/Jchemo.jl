"""
    dmnorm(X = nothing; mu = nothing, S = nothing,
        simpl::Bool = false)
    dmnorm!(X = nothing; mu = nothing, S = nothing,
        simpl::Bool = false)
Normal probability density estimation.
* `X` : X-data (n, p) used to estimate the mean and 
    the covariance matrix. If `nothing`, `mu` and `S` 
    must be provided.
* `mu` : Mean vector of the normal distribution. 
    If `nothing`, `mu` is computed by the column-means of `X`.
* `S` : Covariance matrix of the normal distribution.
    If `nothing`, `S` is computed by cov(`X`; corrected = true).
* `simpl` : Boolean. If `true`, the constant term and the determinant 
    in the density formula are set to 1. Default to `false`.

Data `X` can be univariate (p = 1) or multivariate (p > 1). See examples.

When `simple = true`, the determinant of the covariance matrix (object `detS`) 
and the constant (2 * pi)^(-p / 2) (object `cst`) in the density formula are 
set to 1. The function returns a pseudo density that resumes to exp(-d / 2), 
where d is the squared Mahalanobis distance to the fcenter.

This can for instance be useful when the number of columns (p) of 
`X` becomes too large and when consequently:
* `detS` tends to 0 or, conversely, to infinity
* `cst` tends to 0
which makes impossible to compute the true density. 

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

nlv = 2
fmda = fda(X, y; nlv) ;
pnames(fmda)
T = fmda.T
head(T)
n, p = size(T)

####  Probability density in the FDA score space (2D)

## Example of class Setosa 
s = y .== "setosa"
zT = T[s, :]

## Bivariate distribution
fm = dmnorm(zT) ;
pnames(fm)
fm.Uinv 
fm.detS
pred = Jchemo.predict(fm, zT).pred
head(pred) 

mu = colmean(zT)
S = cov(zT)
dmnorm(; mu = mu, S = S).Uinv
dmnorm(; mu = mu, S = S).detS

npoints = 2^7
lims = [(minimum(zT[:, j]), maximum(zT[:, j])) for j = 1:nlv]
x1 = LinRange(lims[1][1], lims[1][2], npoints)
x2 = LinRange(lims[2][1], lims[2][2], npoints)
z = mpar(x1 = x1, x2 = x2)
grid = reduce(hcat, z)
m = nro(grid)
fm = dmnorm(zT) ;
res = Jchemo.predict(fm, grid) ;
pred_grid = vec(res.pred)
f = Figure(size = (600, 400))
ax = Axis(f[1, 1]; 
    title = "Density for FDA scores (Iris - Setosa)",
    xlabel = "Score 1", ylabel = "Score 2")
co = contour!(ax, grid[:, 1], grid[:, 2], pred_grid; 
    levels = 10, labels = true)
#Colorbar(f[1, 2], co; label = "Density")
scatter!(ax, T[:, 1], T[:, 2],
    color = :red, markersize = 5)
scatter!(ax, zT[:, 1], zT[:, 2],
    color = :blue, markersize = 5)
#xlims!(ax, -12, 12) ;ylims!(ax, -12, 12)
f

## Univariate distribution
j = 1
x = zT[:, j]
fm = dmnorm(x) ;
pred = Jchemo.predict(fm, x).pred 
f = Figure()
ax = Axis(f[1, 1]; 
    xlabel = string("FDA-score ", j))
hist!(ax, x; bins = 30, normalization = :pdf)  # area = 1
scatter!(ax, x, vec(pred);
    color = :red)
f

x = zT[:, j]
npoints = 2^8
lims = [minimum(x), maximum(x)]
#delta = 5 ; lims = [minimum(x) - delta, maximum(x) + delta]
grid = LinRange(lims[1], lims[2], npoints)
fm = dmnorm(x) ;
pred_grid = Jchemo.predict(fm, grid).pred 
f = Figure()
ax = Axis(f[1, 1];
    xlabel = string("FDA-score ", j))
hist!(ax, x; bins = 30, normalization = :pdf)  # area = 1
lines!(ax, grid, vec(pred_grid); 
    color = :red)
f
```
""" 
function dmnorm(X = nothing; mu = nothing, S = nothing,
     simpl::Bool = false)
    isnothing(S) ? zS = nothing : zS = copy(S)
    dmnorm!(X; mu = mu, S = zS, simpl = simpl)
end

function dmnorm!(X = nothing; mu = nothing, S = nothing,
        simpl::Bool = false)
    !isnothing(X) ? X = ensure_mat(X) : nothing
    if isnothing(mu)
        mu = vec(mean(X, dims = 1))
    end
    if isnothing(S)
        S = cov(X; corrected = true)
    end
    U = cholesky!(Hermitian(S)).U # This modifies S only if S is provided
    if simpl 
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
    Dmnorm(mu, U, detS, cst)
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



        
    