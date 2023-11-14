function snipalsmix(X; par = Par())
    X = ensure_mat(X)
    n, p = size(X)
    res = nipals(X; par)
    t = res.u * res.sv
    t0 = similar(X, n)
    v = similar(X, p)
    absv = copy(v)
    cont = true
    iter = 1
    nrm = p - par.nvar
    while cont
        t0 .= copy(t)
        mul!(v, X', t)
        ## Sparsity
        if nrm > 0
            absv .= abs.(v)
            sel = sortperm(absv; rev = true)[1:par.nvar]
            vmax = v[sel]
            v .= zeros(eltype(X), p)
            v[sel] .= vmax
            zdelta = maximum(sort(absv)[1:nrm])
            v .= soft.(v, zdelta)
        end
        ## End
        v ./= norm(v)
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

