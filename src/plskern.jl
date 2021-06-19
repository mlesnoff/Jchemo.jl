struct Plsr
    T::Matrix{Float64}
    P::Matrix{Float64}
    R::Matrix{Float64}
    W::Matrix{Float64}
    C::Matrix{Float64}
    TT::Vector{Float64}
    xmeans::Vector{Float64}
    ymeans::Vector{Float64}
    weights::Vector{Float64}
    ## For consistency with plsrannar
    U::Union{Array{Float64}, Nothing}
end

"""
    plskern(X, Y, weights = ones(size(X, 1)) ; nlv)
    plskern!(X, Y, weights = ones(size(X, 1)) ; nlv)
PLSR "Improved kernel algorithm #1" (Dayal & McGegor, 1997).
- X : matrix (n, p), or vector (n,).
- Y : matrix (n, q), or vector (n,).
- weights : vector (n,).
- nlv : Nb. latent variables (LVs).

X and Y are internally centered. 
The inplace version modifies externally X and Y. 

Dayal, B.S., MacGregor, J.F., 1997. Improved PLS algorithms. Journal of Chemometrics 11, 73-85.
""" 
function plskern(X, Y, weights = ones(size(X, 1)) ; nlv)
    plskern!(copy(X), copy(Y), weights; nlv = nlv)
end

function plskern!(X, Y, weights = ones(size(X, 1)) ; nlv)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)
    ## Initialization
    nlv = min(nlv, n, p)
    weights = mweights(weights)
    xmeans = colmeans(X, weights) 
    ymeans = colmeans(Y, weights)   
    center!(X, xmeans)
    center!(Y, ymeans)
    T = similar(X, n, nlv)
    P = similar(X, p, nlv)
    W = copy(P)
    R = copy(P)
    C = similar(X, q, nlv)
    TT = similar(X, nlv)
    D = LinearAlgebra.Diagonal(weights)
    XtY = X' * (D * Y)                   # = Xd' * Y = X' * D * Y  (Xd = D * X   Very costly!!)
    ## Pre-allocation of the temporary results
    t   = similar(X, n)
    dt  = similar(X, n)   
    zp  = similar(X, p)
    w   = similar(X, p)
    r   = similar(X, p)
    c   = similar(X, q)
    tmp = similar(XtY)
    ## Computations
    @inbounds for a = 1:nlv
        if q == 1
            w .= vcol(XtY, 1)
        else
            tmp .= XtY
            u = svd!(tmp).V           # = svd(XtY').U = eigen(XtY' * XtY).vectors[with ordering]
            mul!(w, XtY, vcol(u, 1))             
        end
        w ./= sqrt(dot(w, w))    # w .= w ./ sqrt(dot(w, w))                            
        r .= w
        if a > 1
            @inbounds for j = 1:(a - 1)
                r .-= dot(w, vcol(P, j)) .* vcol(R, j)    # r .= r .- ...
            end
        end                   
        mul!(t, X, r)                 # t = X * r
        dt .= weights .* t            # dt = D * t
        tt = dot(t, dt)               # tt = t' * dt = t' * D * t 
        mul!(c, XtY', r)
        c ./= tt                      # c = XtY' * r / tt
        mul!(zp, X', dt)              # zp = (D * X)' * t = X' * (D*t)
        XtY .-= mul!(tmp, zp, c')     # XtY = XtY - zp * c'
        P[:, a] .= zp ./ tt    
        T[:, a] .= t
        W[:, a] .= w
        R[:, a] .= r
        C[:, a] .= c
        TT[a] = tt
     end
     Plsr(T, P, R, W, C, TT, xmeans, ymeans, weights, nothing)
end

"""
    summary(object::Plsr, X)
Summarize the maximal (i.e. with maximal nb. LVs) fitted model.
- object : The fitted model.
- X : The X-data that was used to fit the model.
""" 
function Base.summary(object::Plsr, X)
    n, nlv = size(object.T)
    X = center(X, object.xmeans)
    ## Could be center! but changes x
    ## If too heavy ==> Makes summary!
    sstot = sum(object.weights' * (X.^2))
    tt = object.TT
    tt_adj = vec(sum(object.P.^2, dims = 1)) .* tt
    pvar = tt_adj / sstot
    cumpvar = cumsum(pvar)
    xvar = tt_adj / n    
    explvar = DataFrame(nlv = 1:nlv, var = xvar, pvar = pvar, cumpvar = cumpvar) 
    (explvar = explvar,)
end

""" 
    transform(object::Plsr, X; nlv = nothing)
Compute LVs ("scores" T) from a fitted model and a matrix X.
- object : The maximal fitted model.
- X : Matrix (m, p) for which LVs are computed.
- nlv: Nb. LVs to consider. If nothing, it is the maximum nb. LVs.
""" 
function transform(object::Plsr, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    T = center(X, object.xmeans) * @view(object.R[:, 1:nlv])
    ## Could be center! but changes x
    ## If too heavy ==> Makes summary!
    T
end

"""
    coef(object::Plsr; nlv = nothing)
Compute the b-coefficients of a fitted model.
- object : The maximal fitted model.
- nlv: Nb. LVs to consider. If nothing, it is the maximum nb. LVs.

The returned object B is a matrix (p, q). If nlv = 0, B is a matrix of zeros.
""" 
function coef(object::Plsr; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    beta = object.C[:, 1:nlv]'
    B = @view(object.R[:, 1:nlv]) * beta
    int = object.ymeans' .- object.xmeans' * B
    (int = int, B = B)
end


"""
    predict(object::Plsr, X; nlv = nothing)
Compute Y-predictions from a fitted model.
- object : The maximal fitted model.
- X : Matrix (m, p) for which predictions are computed.
- nlv : Nb. LVs, or collection of nb. LVs, to consider. 
If nothing, it is the maximum nb. LVs.
""" 
function predict(object::Plsr, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(le_nlv)
    for i = 1:le_nlv
        z = coef(object ; nlv = nlv[i])
        pred[i] = z.int .+ X * z.B
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end

