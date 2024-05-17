function sampwsp(X, dcri; maxit = nro(X))
    X = ensure_mat(X)
    n, p = size(X)
    indX = collect(1:n)
    indXtot = copy(indX)
    x = similar(X, 1, p)
    ## First reference point: set as the closest 
    ## from the domain center 
    xmeans = colmean(X)
    s = getknn(X, xmeans'; k = 1).ind[1][1]
    x .= vrow(X, s:s)
    ind = [indX[s]]
    ## Start
    iter = 1
    while (n > 1) && (iter < maxit) 
        res = getknn(X, x; k = n)
        s = res.d[1] .> dcri
        v = res.ind[1][s]  # new {reference point + candidates}
        if length(v) > 0
            s1 = v[1]
            s2 = v[2:end]
            x .= vrow(X, s1:s1)  # reference point
            X = vrow(X, s2)      # candidates
            push!(ind, indX[s1])
            indX = indX[s2]
            n = nro(X)
        else
            train = rmrow(indXtot, ind)
            return (train, test = ind, niter = iter)
        end
        iter += 1
    end
    train = rmrow(indXtot, ind)
    (train, test = ind, niter = iter)
end

