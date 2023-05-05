struct Dmnorm
    mu
    Uinv 
    detS
end

"""
    dmnorm(X = nothing; mu = nothing, S = nothing)
Compute normal probability density for multivariate data.
* `X` : X-data used to estimate the mean and 
    the covariance matrix of the population. 
    If `nothing`, `mu` and `S` must be provided.
* `mu` : Mean vector of the normal distribution. 
    If `nothing`, `mu` is computed by the column-means of `X`.
* `S` : Covariance matrix of the normal distribution.
    If `nothing`, `S` is computed by cov(`X`).

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
fmda = fda(X, y; nlv = nlv) ;
pnames(fmda)
T = fmda.T
head(T)
n, p = size(T)

####  Probability density in the FDA score space (2D)


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
    detS = det(U)^2  
    detS == 0 ? detS = 1e-20 : nothing
    Uinv = LinearAlgebra.inv!(U)
    #cholesky!(S)
    #U = sqrt(diag(diag(S), nrow = p))
    #Uinv = solve(diag(diag(S), nrow = p))
    Dmnorm(mu, Uinv, detS)
end

function predict(object::Dmnorm, X)
    X = ensure_mat(X)
    p = size(X, 2)
    mu = reshape(object.mu, 1, length(object.mu))
    d = mahsqchol(X, mu, object.Uinv)
    ds = (2 * pi)^(-p / 2) * (1 / sqrt(object.detS)) * exp.(-.5 * d)
    (pred = ds,)
end



        
    