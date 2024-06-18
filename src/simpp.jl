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
        P = P[:, 1:(n + znsim)]
    end
    fscale!(P, colnorm(P))
end

simppsph = function(X; nsim = 0)
    X = ensure_mat(X)
    n, p = size(X)
    P = similar(X, p, n + nsim)
    P[:, 1:n] .= X'    
    if nsim > 0
        w = similar(X, n)
        for j = (n + 1):(n + nsim)
            w .= rand(Uniform(-1, 1), n)
            P[:, j] .= colsum(X, mweight(w))
        end
    end
    fscale!(P, colnorm(P))
end


