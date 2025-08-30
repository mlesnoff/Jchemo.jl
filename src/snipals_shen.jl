function snipals_shen(X; kwargs...)
    par = recovkw(Jchemo.ParSnipals, kwargs).par 
    X = ensure_mat(X)
    p = nco(X)
    if par.meth == :soft 
        fthresh = thresh_soft
    elseif par.meth == :hard 
        fthresh = thresh_hard
    end 
    nvar = par.nvar
    res = nipals(X; kwargs...)
    u = res.u
    u0 = similar(u)
    vtild = similar(X, p)
    absv = similar(vtild)
    ind = list(Int64, p)
    sel = list(Int64, nvar)
    cont = true
    iter = 1
    nzeros = p - par.nvar  # = degree of sparsity 
    while cont
        u0 .= copy(u)
        mul!(vtild, X', u)
        ## Sparsity
        if nzeros > 0
            absv .= abs.(vtild)
            ind .= sortperm(absv; rev = true)
            sel .= ind[1:nvar]
            qt = minimum(absv[sel])
            lambda = maximum(absv[absv .< qt])
            vtild .= fthresh.(vtild, lambda)
        end
        ## End
        mul!(u, X, vtild)
        u ./= normv(u)
        dif = sum((u .- u0).^2)
        iter = iter + 1
        if (dif < par.tol) || (iter > par.maxit)
            cont = false
        end
    end
    niter = iter - 1
    v = vtild / normv(vtild)
    t = X * v 
    (t = t, v, vtild, niter)
end

