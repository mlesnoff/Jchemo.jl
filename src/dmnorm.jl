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



        
    