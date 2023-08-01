function snipalsh(X; nvar = nco(X),
        tol = sqrt(eps(1.)), maxit = 200)
    X = ensure_mat(X)
    n, p = size(X)
    res = nipals(X; tol = tol, maxit = maxit)
    t = res.u * res.sv
    t0 = similar(X, n)
    v = similar(X, p)
    absv = copy(v)
    cont = true
    iter = 1
    while cont
        t0 .= copy(t)
        mul!(v, X', t)
        ## Sparsity
        absv .= abs.(v)
        sel = sortperm(absv; rev = true)[1:nvar]
        vmax = v[sel]
        v .= zeros(p)
        v[sel] .= vmax
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

