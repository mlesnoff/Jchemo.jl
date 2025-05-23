## Simulation of PP directions (normed to 1) from X-data (n, p).
## The first n directions are the rows of X. 

## Hubert, M., Rousseeuw, V.J., Vanden Branden, K., 2005. ROBPCA: 
## A New Approach to Robust Principal Component Analysis. 
## Technometrics 47, 64-79. https://doi.org/10.1198/004017004000000563.
## The directions go through the two points of pairs of observations.
simpphub = function(X; nsim = 0, cst = 50)
    X = ensure_mat(X)
    n, p = size(X)
    V = similar(X, p, n + nsim)
    V[:, 1:n] .= X'
    if nsim > 0
        s1 = rand(1:n, cst * nsim)
        s2 = rand(1:n, cst * nsim)
        u = (s1 .- s2) .!= 0
        S = hcat(s1[u], s2[u])
        S = unique(S; dims = 1)
        znsim = min(nsim, nro(S))
        k = 1
        for j = (n + 1):(n + znsim)
            V[:, j] .= vrow(X, S[k, 1]) - vrow(X, S[k, 2])
            k += 1
        end
        V = vcol(V, 1:(n + znsim))
    end
    fscale!(V, colnorm(V))
    V
end

## Sphere
simppsph = function(X; nsim = 0)
    X = ensure_mat(X)
    n, p = size(X)
    V = similar(X, p, n + nsim)
    V[:, 1:n] .= X'    
    if nsim > 0
        w = similar(X, n)
        for j = (n + 1):(n + nsim)
            w .= rand(Uniform(-1, 1), n)
            V[:, j] .= colsum(X, mweight(w))
        end
    end
    fscale!(V, colnorm(V))
    V
end


