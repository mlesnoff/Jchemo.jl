simppbin = function(X; nsim = 0)
    X = ensure_mat(X)
    n, p = size(X)
    P = similar(X, p, n + nsim)
    P[:, 1:n] .= X'
    if nsim > 0
        for j = (n + 1):(n + nsim)
            P[:, j] .= rand(0:1, p)
        end
    end
    fscale!(P, colnorm(P))
end

simpphub = function(X; nsim = 0, cst = 50)
    X = ensure_mat(X)
    n, p = size(X)
    P = similar(X, p, n + nsim)
    P[:, 1:n] .= X'
    if nsim > 0
        s1 = rand(1:n, cst * nsim)
        s2 = rand(1:n, cst * nsim)
        u = (s1 .- s2) .!= 0
        S = hcat(s1[u], s2[u])
        S = unique(S; dims = 1)
        znsim = min(nsim, nro(S))
        k = 1
        for j = (n + 1):(n + znsim)
            P[:, j] .= X[S[k, 1], :] - X[S[k, 2], :]
            k += 1
        end
    end
    fscale!(P, colnorm(P))
end

simpplc = function(X; nsim = 0)
    X = ensure_mat(X)
    n, p = size(X)
    P = similar(X, p, n + nsim)
    P[:, 1:n] .= X'
    if nsim > 0
        for j = (n + 1):(n + nsim)
            w = mweight(rand(n))
            P[:, j] .= colsum(X, w)
        end
    end
    fscale!(P, colnorm(P))
end

simppsph = function(X; nsim = 0)
    X = ensure_mat(X)
    n, p = size(X)
    P = similar(X, p, n + nsim)
    P[:, 1:n] .= X'
    if nsim > 0
        z = similar(X, p)
        for j = (n + 1):(n + nsim)
            z .= rand(Uniform(-1, 1), p)
            P[:, j] .= z / norm(z)
        end
    end
    fscale!(P, colnorm(P))
end





