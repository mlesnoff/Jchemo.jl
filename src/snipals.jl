function snipals(X; kwargs...)
    par = recovkw(ParSnipals, kwargs).par 
    X = ensure_mat(X)
    p = nco(X)
    nvar = par.nvar
    if par.meth == :soft 
        fun = thresh_soft
    elseif par.meth == :hard 
        fun = thresh_hard
    end 
    res = nipals(X; kwargs...)
    t = res.u * res.sv
    t0 = similar(t)
    v = similar(res.v)
    v0 = similar(v)
    absv = copy(v)
    u = list(Int64, p)
    sel = list(Int64, nvar)
    cont = true
    iter = 1
    nzeros = p - nvar  # degree of sparsity
    while cont
        v0 .= copy(v)
        t0 .= copy(t)
        mul!(v, X', t)
        ## Sparsity
        if nzeros > 0
            absv .= abs.(v)
            u .= sortperm(absv; rev = true)
            sel .= u[1:nvar]
            qt = minimum(absv[sel])
            lambda = maximum(absv[absv .< qt])
            v .= fun.(v, lambda)
        end
        ## End
        v ./= normv(v)
        mul!(t, X, v)
        difv = sum((v .- v0).^2)
        dift = sum((t .- t0).^2)
        iter = iter + 1
        if (difv < par.tol) || (dift < par.tol) || (iter > par.maxit)
            cont = false
        end
    end
    niter = iter - 1
    (t = t, v, niter)
end

