Base.@kwdef mutable struct JchemoModel1{T <: Function, K <: Base.Pairs}
    algo::T   
    fm
    kwargs::K
end

function fit!(model::Jchemo.JchemoModel1, X)
    kwargs = values(model.kwargs)
    model.fm = model.algo(X; kwargs...)
    return
end  
function fit!(model::Jchemo.JchemoModel1, X, Y)
    kwargs = values(model.kwargs)
    model.fm = model.algo(X, Y; kwargs...)
    return
end  
function fit!(model::Jchemo.JchemoModel1, X, Y, weights::Weight)
    kwargs = values(model.kwargs)
    model.fm = model.algo(X, Y, weights; kwargs...)
    return
end

plskern06() = FunX{Function}(plskern06)  # for pip

function plskern06(; kwargs...)
    JchemoModel1(plskern06, nothing, kwargs)
end

function plskern06(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    plskern06(X, Y, weights; kwargs...)
end

function plskern06(X, Y, weights::Weight; kwargs...)
    plskern06!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function plskern06!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParPlsr, kwargs).par
    Q = eltype(X)
    n, p = size(X)
    q = nco(Y)
    nlv = min(n, p, maximum(par.nlv)) # 'maximum' required for plsravg 
    xmeans = colmean(X, weights) 
    ymeans = colmean(Y, weights)  
    xscales = ones(Q, p)
    yscales = ones(Q, q)
    if par.scal 
        xscales .= colstd(X, weights)
        yscales .= colstd(Y, weights)
        fcscale!(X, xmeans, xscales)
        fcscale!(Y, ymeans, yscales)
    else
        fcenter!(X, xmeans)
        fcenter!(Y, ymeans)
    end
    D = Diagonal(weights.w)
    XtY = X' * (D * Y)                   # = Xd' * Y = X' * D * Y  (Xd = D * X   Very costly!!)
    #XtY = X' * (weights.w .* Y)         # Can create OutOfMemory errors for very large matrices
    ## Pre-allocation
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
    tmpXtY = similar(XtY) # = XtY_approx
    # End
    @inbounds for a = 1:nlv
        if q == 1
            w .= vcol(XtY, 1)
            w ./= norm(w)
        else
            w .= svd(XtY).U[:, 1]
        end                                  
        r .= w
        if a > 1
            @inbounds for j = 1:(a - 1)
                r .-= dot(w, vcol(P, j)) .* vcol(R, j)    
            end
        end                   
        mul!(t, X, r)                 # t = X * r
        dt .= weights.w .* t            # dt = D * t
        tt = dot(t, dt)               # tt = t' * dt = t' * D * t 
        mul!(c, XtY', r)
        c ./= tt                      # c = XtY' * r / tt
        mul!(zp, X', dt)              # zp = (D * X)' * t = X' * (D * t)
        XtY .-= mul!(tmpXtY, zp, c')     # XtY = XtY - zp * c' ; deflation of the kernel matrix 
        P[:, a] .= zp ./ tt           # ==> the metric applied to covariance is applied outside the loop,
        T[:, a] .= t                  # conversely to other algorithms such as nipals
        W[:, a] .= w
        R[:, a] .= r
        C[:, a] .= c
        TT[a] = tt
    end
    Plsr(T, P, R, W, C, TT, xmeans, xscales, ymeans, yscales, weights, nothing, par)
end

function transf(object::Union{Plsr, Splsr}, X; nlv = nothing)
    X = ensure_mat(X)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    T = fcscale(X, object.xmeans, object.xscales) * vcol(object.R, 1:nlv)
    # Could be fcscale! but changes X
    # If too heavy ==> Makes summary!
    T
end

function coef(object::Union{Plsr, Pcr, Splsr}; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    beta = vcol(object.C, 1:nlv)'
    W = Diagonal(object.yscales)
    B = Diagonal(1 ./ object.xscales) * vcol(object.R, 1:nlv) * beta * W
    ## 'int': No correction is needed, since 
    ## ymeans, xmeans and B are in the original scale 
    int = object.ymeans' .- object.xmeans' * B
    (B = B, int = int)
end

function predict(object::Union{Plsr, Pcr, Splsr}, X; nlv = nothing)
    X = ensure_mat(X)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = (max(0, minimum(nlv)):min(a, maximum(nlv)))
    le_nlv = length(nlv)
    pred = list(Matrix{eltype(X)}, le_nlv)
    @inbounds  for i = 1:le_nlv
        z = coef(object; nlv = nlv[i])
        pred[i] = z.int .+ X * z.B
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end

function Base.summary(object::Union{Plsr, Splsr}, X)
    X = ensure_mat(X)
    n, nlv = size(object.T)
    X = fcscale(X, object.xmeans, object.xscales)
    ## Could be fcscale! but changes X
    ## If too heavy ==> Makes summary!
    sstot = sum(object.weights.w' * (X.^2)) # = frob(X, object.weights)^2 
    tt = object.TT
    tt_adj = colsum(object.P.^2) .* tt      # tt_adj[a] = p[a]'p[a] * tt[a]
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvarx = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar)     
    (explvarx = explvarx,)
end
