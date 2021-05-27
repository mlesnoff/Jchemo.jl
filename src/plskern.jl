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
    plskern!(X, Y, weights = ones(size(X, 1)) ; nlv)
PLSR "Improved kernel algorithm #1" 
Dayal, B.S., MacGregor, J.F., 1997. Improved PLS algorithms. Journal of Chemometrics 11, 73-85.
- X {Float64}: matrix (n, p) with p >= 1, or vector (n,)
- Y {Float64}: matrix (n, q) with q >= 1, or vector (n,)
- weights: vector (n,)
- nlv: Integer > 0
X and Y are internally centered.
For saving allocation memory, the centering is done "inplace",
which modifies externally X and Y. 
""" 
function plskern!(X, Y, weights = ones(size(X, 1)) ; nlv)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)
    @assert size(Y, 1) == n "X and Y do not have the same nb. observations."
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
    XtY = X' * (D * Y)  
    ## XtY = Xd' * Y = X' * D * Y  (Xd = D * X   Very costly!!)
    ## Pre-allocation of the temporary results
    t   = similar(X, n)
    dt  = similar(X, n)   
    zp  = similar(X, p)
    w   = similar(X, p)
    r   = similar(X, p)
    c   = similar(X, q)
    tmp = similar(XtY)
    ## Computations
    for a = 1:nlv
        if q == 1
            w .= col(XtY, 1)
        else
            tmp .= XtY
            u = svd!(tmp).V   
            ## = svd(XtY').U = eigen(XtY' * XtY).vectors[with ordering]
            mul!(w, XtY, col(u, 1))             
        end
        w ./= sqrt(dot(w, w))
        ## w .= w ./ ...                            
        r .= w
        if a > 1
            @inbounds for j = 1:(a - 1)
                r .-= dot(w, col(P, j)) .* col(R, j)
                ## r .= r .- ...
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
    plskern(X, Y, weights = ones(size(X, 1)) ; nlv)
Makes a preliminary copy of X and Y and then runs plskern! on these copies.
Inputs X and Y are not modified.
""" 
function plskern(X, Y, weights = ones(size(X, 1)) ; nlv)
    res = plskern!(copy(X), copy(Y), weights ; nlv)
    res
end

####################################### AUXILIARY

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

function transform(object::Plsr, X ; nlv::Union{Int64, Nothing} = nothing)
    a = size(object.T, 2)
    nlv == nothing ? nlv = a : nlv = min(nlv, a)
    T = center(X, object.xmeans) * @view(object.R[:, 1:nlv])
    ## Could be center! but changes x
    ## If too heavy ==> Makes summary!
    T
end

function coef(object::Plsr ; nlv::Union{Int64, Nothing} = nothing)
    a = size(object.T, 2)
    nlv == nothing ? nlv = a : nlv = min(nlv, a)
    beta = @view(object.C[:, 1:nlv])'
    B = @view(object.R[:, 1:nlv]) * beta
    int = object.ymeans' .- object.xmeans' * B
    (int = int, B = B)
end

"""
predict(object::Plsr, X ; nlv::Union{Int64, Nothing} = nothing)
Works also for nlv = 0, 
since coef() returns a matrix of zeros for B
"""
function predict(object::Plsr, X ; nlv::Union{Int64, UnitRange{Int64}, Array{Int64}, Nothing} = nothing)
    a = size(object.T, 2)
    nlv == nothing ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(le_nlv)
    for i = 1:le_nlv
        z = coef(object ; nlv = nlv[i])
        pred[i] = z.int .+ X * z.B
    end 
    if le_nlv == 1
        pred = pred[1]
    end
    (pred = pred,)
end

"""
predict_beta(object::Pls, X ; nlv::Union{Int64, Nothing} = nothing)
Works also for nlv = 0
"""
function predict_beta(object::Plsr, X ; nlv::Union{Int64, Nothing} = nothing)
    a = size(object.T, 2)
    nlv == nothing ? nlv = a : nlv = min(nlv, a)
    beta = @view(object.C[:, 1:nlv])'
    pred = object.ymeans' .+ transform(object, X ; nlv) * beta
    (pred = pred,)
end
    

