function dmnorm(X = NULL; mu = nothing, sigma = nothing)
    X = ensure_mat(X)
    isnothing(mu) ? mu = means(X, dims = 1) : nothing
    isnothing(sigma) ? sigma = cov(X) : nothing

    U = chol(sigma)
    Uinv = inv(U)

    #U = sqrt(diag(diag(sigma), nrow = p))
    #Uinv = solve(diag(diag(sigma), nrow = p))
     
    zdet = det(U)^2
    if zdet == 0 
        zdet = 1e-20
    end
    (mu = mu, Uinv = Uinv, det = zdet)
end

function predict(object::Dmnorm, X) 
    X = ensure_mat(X)
    ## squared distance
    d = mahsq_mu(X, mu = object.mu, Uinv = object.Uinv)
    ## density
    ds = (2 * pi)^(-p / 2) * (1 / sqrt(object.det)) * exp(-.5 * d)
    (pred = ds,)
end

        
    