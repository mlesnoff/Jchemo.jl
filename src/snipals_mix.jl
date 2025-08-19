## Returns the same results as snipals_shen
function snipals_mix(X; kwargs...)
    par = recovkw(ParSnipals, kwargs).par 
    X = ensure_mat(X)
    p = nco(X)
    if par.meth == :soft 
        fthresh = thresh_soft
    elseif par.meth == :hard 
        fthresh = thresh_hard
    end 
    nvar = par.nvar
    res = nipals(X; kwargs...)
    t = res.t
    v = similar(res.v)
    v0 = similar(v)
    absv = copy(v)
    ind = list(Int64, p)
    sel = list(Int64, nvar)
    cont = true
    iter = 1
    nzeros = p - nvar  # degree of sparsity
    while cont
        v0 .= copy(v)
        mul!(v, X', t)
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
        v ./= normv(v)
        mul!(t, X, v)
        dif = sum((v .- v0).^2)
        iter = iter + 1
        if (dif < par.tol) || (iter > par.maxit)
                cont = false
        end
    end
    niter = iter - 1
    (t = t, v, niter)
end

