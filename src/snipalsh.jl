function snipalsh(X; nvar = nco(X),
        tol = sqrt(eps(1.)), maxit = 200)
    X = ensure_mat(X)
    n, p = size(X)
    u = nipals(X; tol = tol, maxit = maxit).u
    u0 = similar(X, n)
    v = similar(X, p)
    absv = copy(v)
    cont = true
    iter = 1
    #s = 0
    while cont
        u0 .= copy(u)
        mul!(v, X', u)
        ## Sparsity
        absv .= abs.(v)
        sel = sortperm(absv; rev = true)[1:nvar]
        vmax = v[sel]
        v .= zeros(p)
        v[sel] .= vmax
        ## End
        mul!(u, X, v)
        #s = norm(u)
        #u ./= s
        u ./= norm(u)
        v ./= norm(v)
        dif = sum((u .- u0).^2)
        iter = iter + 1
        if (dif < tol) || (iter > maxit)
            cont = false
        end
    end
    #sv = sqrt(s)
    niter = iter - 1
    #(u = u, v, sv, niter)
    (v = v, niter)
end

