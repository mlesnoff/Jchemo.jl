function snipals_soft(X; kwargs...)
    par = recovkw(ParSnipals, kwargs).par 
    X = ensure_mat(X)
    Q = eltype(X)
    n, p = size(X)
    res = nipals(X; kwargs...)
    t = res.u * res.sv
    t0 = similar(X, n)
    v = similar(X, p)
    absv = copy(v)
    cont = true
    iter = 1
    nzeros = p - par.nvar
    while cont
        t0 .= copy(t)
        mul!(v, X', t)
        ## Sparsity
        if nzeros > 0
            absv .= abs.(v)
            sel = sortperm(absv; rev = true)[1:par.nvar]
            vhigh = v[sel]
            v .= zeros(Q, p)
            v[sel] .= vhigh
            qt = minimum(abs.(vhigh))
            delta = maximum(absv[absv .< qt])
            v .= soft.(v, delta)
        end
        ## End
        v ./= normv(v)
        mul!(t, X, v)
        dif = sum((t .- t0).^2)
        iter = iter + 1
        if (dif < par.tol) || (iter > par.maxit)
            cont = false
        end
    end
    niter = iter - 1
    (t = t, v, niter)
end

