function splsrh(X, Y, weights = ones(nro(X)); nlv, 
        nvar = 1, scal::Bool = false)
    splsrh!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; 
        nlv = nlv, nvar = nvar, scal = scal)
end

function splsrh!(X::Matrix, Y::Matrix, weights = ones(nro(X)); 
        nlv, nvar = 1, scal::Bool = false)
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, nlv)
    weights = mweight(weights)
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)  
    xscales = ones(p)
    yscales = ones(q)
    if scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        cscale!(X, xmeans, xscales)
        cscale!(Y, ymeans, yscales)
    else
        center!(X, xmeans)
        center!(Y, ymeans)
    end
    D = Diagonal(weights)
    XtY = X' * (D * Y)                   # = Xd' * Y = X' * D * Y  (Xd = D * X   Very costly!!)
    #XtY = X' * (weights .* Y)           # Can create OutOfMemory errors for very large matrices
    # Pre-allocation
    T = similar(X, n, nlv)
    W = similar(X, p, nlv)
    P = copy(W)
    R = copy(P)
    C = similar(X, q, nlv)
    TT = similar(X, nlv)
    t   = similar(X, n)
    dt  = similar(X, n)   
    zp  = similar(X, p)
    w   = similar(X, p)
    r   = similar(X, p)
    c   = similar(X, q)
    tmp = similar(XtY) # = XtY_approx
    sellv = list(nlv, Vector{Int64})
    w2 = copy(w)
    # End
    @inbounds for a = 1:nlv
        if q == 1
            w .= vcol(XtY, 1)
            w ./= norm(w)
        else
            w .= svd(XtY).U[:, 1]
        end
        w2 .= w.^2
        sellv[a] = sortperm(w2; rev = true)[1:nvar]
        wmax = w[sellv[a]]
        w .= zeros(p)
        w[sellv[a]] .= wmax
        r .= w
        if a > 1
            @inbounds for j = 1:(a - 1)
                r .-= dot(w, vcol(P, j)) .* vcol(R, j)    
            end
        end                   
        mul!(t, X, r)                 # t = X * r
        dt .= weights .* t            # dt = D * t
        tt = dot(t, dt)               # tt = t' * dt = t' * D * t 
        mul!(c, XtY', r)
        c ./= tt                      # c = XtY' * r / tt
        mul!(zp, X', dt)              # zp = (D * X)' * t = X' * (D * t)
        XtY .-= mul!(tmp, zp, c')     # XtY = XtY - zp * c' ; deflation of the kernel matrix 
        P[:, a] .= zp ./ tt           # ==> the metric applied to covariance is applied outside the loop,
        T[:, a] .= t                  # conversely to other algorithms such as nipals
        W[:, a] .= w
        R[:, a] .= r
        C[:, a] .= c
        TT[a] = tt
     end
     sel = reduce(vcat, sellv)
     Splsr1(T, P, R, W, C, TT, xmeans, xscales, ymeans, 
         yscales, weights, sellv, sel)
end

