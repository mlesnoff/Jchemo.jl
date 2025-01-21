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
    v = similar(X, p) # = 'v_tild' in Shen et al. 2008
    absv = similar(v)
    ind = list(Int64, p)
    sel = list(Int64, nvar)
    cont = true
    iter = 1
    nzeros = p - par.nvar  # = degree of sparsity 
    while cont
        u0 .= copy(u)
        mul!(v, X', u)
        ## Sparsity
        if nzeros > 0
            absv .= abs.(v)
            ind .= sortperm(absv; rev = true)
            sel .= ind[1:nvar]
            qt = minimum(absv[sel])
            lambda = maximum(absv[absv .< qt])
            v .= fthresh.(v, lambda)
        end
        ## End
        mul!(u, X, v)
        u ./= normv(u)
        dif = sum((u .- u0).^2)
        iter = iter + 1
        if (dif < par.tol) || (iter > par.maxit)
            cont = false
        end
    end
    niter = iter - 1
    vtild = copy(v)
    v ./= normv(v)  # final unitary vector 'v'
    t = X * v 
    (t = t, v, vtild, niter)
end

