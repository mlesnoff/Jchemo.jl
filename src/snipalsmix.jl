function snipalsmix(X; nvar = nco(X),
        tol = sqrt(eps(1.)), maxit = 200)
    X = ensure_mat(X)
    n, p = size(X)
    res = nipals(X; tol = tol, maxit = maxit)
    u = res.u
    u0 = similar(X, n)
    v = similar(X, p)
    absv = copy(v)
    cont = true
    iter = 1
    sv = 0
    while cont
        u0 .= copy(u)
        mul!(v, X', u)
        ## Sparsity
        absv .= abs.(v)
        sellv[a] = sortperm(absv; rev = true)[1:znvar]
        vmax = v[sellv[a]]
        v .= zeros(p)
        v[sellv[a]] .= vmax
        delta = maximum(sort(absv)[1:nrm])
        v .= soft.(v, delta)
        ## End
        mul!(u, X, v)
        sv = norm(u)
        u ./= sv
        v ./= norm(v)
        dif = sum((u .- u0).^2)
        iter = iter + 1
        if (dif < tol) || (iter > maxit)
            cont = false
        end
    end
    niter = iter - 1
    (u = u, v, sv, niter)
end

