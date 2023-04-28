struct Dmnorm
    mu
    Uinv 
    det
end

"""
    dmnorm(X = nothing; mu = nothing, S = nothing)
Compute the normal probability density of 
multivariate observations.
* `X` : X-data used to estimate the mean and 
    the covariance matrix of the population. 
    If `nothing`, `mu` and `S` must be provided.
* `mu` : Mean vector of the normal distribution. 
    If `nothing`, `mu` is computed by the column-means of `X`.
* `S` : Covariance matrix of the normal distribution.
    If `nothing`, `S` is computed by cov(`X`).

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/iris.jld2") 
@load db dat
pnames(dat)
X = dat.X 
summ(X)

## Studying a two-dimensionnal distribution
## of the class "Setosa"
Xtrain = Matrix(X[1:40, 1:2])
Xtest = Matrix(X[41:50, 1:2])

fm = dmnorm(Xtrain) ;
fm.mu
fm.Uinv
fm.det
Jchemo.predict(fm, Xtest).pred

mu = colmean(Xtrain)
S = cov(Xtrain)
fm = dmnorm(; mu = mu, S = S)
fm.Uinv
fm.det

k = 50
x1 = range(.9 * minimum(Xtrain[:, 1]), 1.1 * maximum(Xtrain[:, 1]), length = k) 
x2 = range(.9 * minimum(Xtrain[:, 2]), 1.1 * maximum(Xtrain[:, 2]), length = k) 
z = reduce(hcat, mpar(x1 = x1, x2 = x2))
pred_train = Jchemo.predict(fm, z).pred
f = Figure()
ax = Axis(f[1, 1], title = "Dmnorm - Setosa",
    xlabel = "Sepal length", ylabel = "Sepal width") ;
co = contour!(ax, z[:, 1], z[:, 2], vec(pred_train))
#co = contourf!(ax, z[:, 1], z[:, 2], vec(zpred), levels = 10)
pred = Jchemo.predict(fm, Xtest).pred
scatter!(ax, Xtest[:, 1], Xtest[:, 2], vec(pred), color = :red)
#xlims!(ax, 3.5, 7) ; ylims!(ax, 2, 5)
Colorbar(f[1, 2], co)
f
```
""" 
function dmnorm(X = nothing; mu = nothing, S = nothing)
    isnothing(S) ? zS = nothing : zS = copy(S)
    dmnorm!(X; mu = mu, S = zS)
end

function dmnorm!(X = nothing; mu = nothing, S = nothing)
    !isnothing(X) ? X = ensure_mat(X) : nothing
    if isnothing(mu)
        mu = vec(mean(X, dims = 1))
    end
    if isnothing(S)
        S = cov(X)
    end
    U = cholesky!(Hermitian(S)).U # This modifies S only if S is provided
    zd = det(U)^2  
    zd == 0 ? zd = 1e-20 : nothing
    Uinv = LinearAlgebra.inv!(U)
    #cholesky!(S)
    #U = sqrt(diag(diag(S), nrow = p))
    #Uinv = solve(diag(diag(S), nrow = p))
    Dmnorm(mu, Uinv, zd)
end

function predict(object::Dmnorm, X)
    X = ensure_mat(X)
    p = size(X, 2)
    mu = reshape(object.mu, 1, length(object.mu))
    d = mahsqchol(X, mu, object.Uinv)
    ds = (2 * pi)^(-p / 2) * (1 / sqrt(object.det)) * exp.(-.5 * d)
    (pred = ds,)
end



        
    