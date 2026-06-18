## Here, sparseness is applied after convergence of Nipals
function snipals_post(X::Matrix{Q}; kwargs...) where Q <: AbstractFloat
    par = recovkw(Jchemo.ParSnipals{Q}, kwargs).par 
    X = ensure_mat(X)
    p = nco(X)
    if par.meth == :soft 
        fthresh = thresh_soft
    elseif par.meth == :hard 
        fthresh = thresh_hard
    end 
    nvar = par.nvar
    res = nipals(X; kwargs...)
    v = res.v
    nzeros = p - par.nvar  # = degree of sparsity 
    ## Sparsity
    if nzeros > 0
        absv = abs.(v)
        ind = sortperm(absv; rev = true)
        sel = ind[1:nvar]
        qt = minimum(absv[sel])
        lambda = maximum(absv[absv .< qt])
        v .= fthresh.(v, lambda)
    end
    ## End
    v ./= normv(v)
    (t = X * v, v, niter = res.niter)
end

