function snipals(X; kwargs...)
    X = ensure_mat(X)
    n, p = size(X)
    res = nipals(X; values(kwargs)...)
    t = res.u * res.sv
    t0 = similar(X, n)
    v = similar(X, p)
    absv = copy(v)
    absv_stand = copy(v)
    theta = copy(v)
    cont = true
    iter = 1
    while cont
        t0 .= copy(t)
        mul!(v, X', t)
        ## Sparsity
        absv .= abs.(v)
        absv_max = maximum(absv)
        absv_stand .= absv / absv_max
        theta .= max.(0, absv_stand .- par.delta) 
        v .= sign.(v) .* theta * absv_max 
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

