function snipals(X; delta = 0,
        tol = sqrt(eps(1.)), maxit = 200)
    X = ensure_mat(X)
    n, p = size(X)
    res = nipals(X; tol = tol, maxit = maxit)
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
        theta .= max.(0, absv_stand .- delta) 
        v .= sign.(v) .* theta * absv_max 
        ## End
        v ./= norm(v)
        mul!(t, X, v)
        dif = sum((t .- t0).^2)
        iter = iter + 1
        if (dif < tol) || (iter > maxit)
            cont = false
        end
    end
    niter = iter - 1
    (t = t, v, niter)
end

