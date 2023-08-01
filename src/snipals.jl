function snipals(X; delta = 0,
        tol = sqrt(eps(1.)), maxit = 200)
    X = ensure_mat(X)
    n, p = size(X)
    u = nipals(X; tol = tol, maxit = maxit).u
    u0 = similar(X, n)
    v = similar(X, p)
    absv = copy(v)
    absv_stand = copy(v)
    theta = copy(v)
    cont = true
    iter = 1
    #s = 0
    while cont
        u0 .= copy(u)
        mul!(v, X', u)
        ## Sparsity
        absv .= abs.(v)
        absv_max = maximum(absv)
        absv_stand .= absv / absv_max
        theta .= max.(0, absv_stand .- delta) 
        v .= sign.(v) .* theta * absv_max 
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
end

