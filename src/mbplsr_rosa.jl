struct MbplsrRosa
    T::Matrix{Float64}
    P::Matrix{Float64}
    R::Matrix{Float64}
    W::Matrix{Float64}
    C::Matrix{Float64}
    TT::Vector{Float64}
    xmeans::Vector
    ymeans::Vector{Float64}
    weights::Vector{Float64}
    bl::Vector
end

"""
    rosaplsr(X, Y, weights = ones(size(X, 1)); nlv)
Multi-block PLSR with the ROSA algorithm (Liland et al. 2016).
* `X` : List (vector) of blocks (matrices) of X-data. 
    Each component of the list is a block.
* `Y` : Y-data.
* `weights` : Weights of the observations (rows).
* `nlv` : Nb. latent variables (LVs) to consider.

The function has the following differences with the original 
algorithm of Liland et al. (2016):
* Scores T are not normed to 1.
* Multivariate `Y` is allowed. In such a case, , the squared residuals are summed 
    over the columns for finding the winning blocks the entered (therefore Y-columns 
    should have the same scale).

`weights` is internally normalized to sum to 1. 

`X` and `Y` are internally centered. 

## References
Liland, K.H., Næs, T., Indahl, U.G., 2016. ROSA—a fast extension of partial least 
squares regression for multiblock data analysis. Journal of Chemometrics 30, 
651–662. https://doi.org/10.1002/cem.2824

## Examples
```julia
using JLD2
mypath = joinpath(@__DIR__, "..", "data")
db = string(mypath, "\\", "ham.jld2") 
@load db dat
pnames(dat) 

X = dat.X
y = dat.Y.c1
group = dat.group
listbl = [1:11, 12:19, 20:25]
X_bl = mblock(X, listbl)
# "New" = first two rows of X_bl 
X_bl_new = mblock(X[1:2, :], listbl)

nlv = 5
fm = mbplsr_rosa(X_bl, y; nlv = nlv) ;
pnames(fm)
fm.T
transform(fm, X_bl_new)
[y predict(fm, X_bl).pred]
predict(fm, X_bl_new).pred
```
""" 
function mbplsr_rosa(X, Y, weights = ones(size(X[1], 1)); nlv)
    nbl = length(X)
    zX = list(nbl, Matrix{Float64})
    @inbounds for k = 1:nbl
        zX[k] = copy(ensure_mat(X[k]))
    end
    mbplsr_rosa!(zX, copy(Y), weights; nlv = nlv)
end

function mbplsr_rosa!(X, Y, weights = ones(size(X[1], 1)); nlv)
    Y = ensure_mat(Y)
    n = size(X[1], 1)
    q = size(Y, 2)   
    nbl = length(X)
    weights = mweight(weights)
    D = Diagonal(weights)
    xmeans = list(nbl, Vector{Float64})
    p = fill(0, nbl)
    @inbounds for k = 1:nbl
        p[k] = size(X[k], 2)
        xmeans[k] = colmean(X[k], weights)   
        center!(X[k], xmeans[k])
    end
    ymeans = colmean(Y, weights)   
    center!(Y, ymeans)
    # Pre-allocation
    W = similar(X[1], sum(p), nlv)
    P = copy(W)
    T = similar(X[1], n, nlv)
    TT = similar(X[1], nlv)    
    C = similar(X[1], q, nlv)
    DY = similar(X[1], n, q)
    t   = similar(X[1], n)
    dt  = similar(X[1], n)   
    c   = similar(X[1], q)
    zp_bl = list(nbl, Vector{Float64})
    zp = similar(X[1], sum(p))
    ssr = similar(X[1], nbl)
    W_bl = list(nbl, Array{Float64})
    w_bl = list(nbl, Vector{Float64})  # List of the weights "w" by block for a given "a"
    zT = similar(X[1], n, nbl)         # Matrix gathering the nbl scores for a given "a"
    bl = fill(0, nlv)
    Res = zeros(n, q, nbl)
    ### Start 
    @inbounds for a = 1:nlv
        DY .= D * Y  # apply the metric on covariance
        @inbounds for k = 1:nbl
            XtY = X[k]' * DY
            if q == 1
                w_bl[k] = vec(XtY)
                w_bl[k] ./= norm(w_bl[k])
            else
                w_bl[k] = svd!(XtY).U[:, 1]
            end
            zT[:, k] .= X[k] * w_bl[k]
        end
        # GS Orthogonalization of the scores
        if a > 1
            z = vcol(T, 1:(a - 1))
            zT .= zT .- z * inv(z' * (D * z)) * z' * (D * zT)
        end
        # Selection of the winner block
        @inbounds for k = 1:nbl
            t = vcol(zT, k)
            dt .= weights .* t
            tt = dot(t, dt)
            Res[:, :, k] .= Y .- (t * t') * DY / tt
        end
        ssr = vec(sum(Res.^2, dims = (1, 2)))
        opt = findmin(ssr)[2][1]
        bl[a] = opt
        # Outputs for winner
        t .= zT[:, opt]
        dt .= weights .* t
        tt = dot(t, dt)
        mul!(c, Y', dt)
        c ./= tt     
        T[:, a] .= t
        TT[a] = tt
        C[:, a] .= c
        Y .= Res[:, :, opt]
        for k = 1:nbl ; zp_bl[k] = X[k]' * dt ; end
        zp .= reduce(vcat, zp_bl)
        P[:, a] .= zp / tt
        # Orthogonalization of the weights "w" by block
        zw = w_bl[opt]
        if (a > 1) && isassigned(W_bl, opt)       
            zW = W_bl[opt]
            zw .= zw .- zW * (zW' * zw)
        end
        zw ./= norm(zw)
        if !isassigned(W_bl, opt) 
            W_bl[opt] = reshape(zw, :, 1)
        else
            W_bl[opt] = hcat(W_bl[opt], zw)
        end
        # Build the weights over the overall matrix
        z = zeros(nbl) ; z[opt] = 1
        W[:, a] .= reduce(vcat, z .* w_bl)
    end
    R = W * inv(P' * W)
    MbplsrRosa(T, P, R, W, C, TT, xmeans, ymeans, weights, bl)
end

""" 
    transform(object::MbplsrRosa, X; nlv = nothing)
Compute LVs ("scores" T) from a fitted model.
* `object` : The maximal fitted model.
* `X` : A list (vector) of blocks (matrices) of X-data for which LVs are computed.
* `nlv` : Nb. LVs to consider. If nothing, it is the maximum nb. LVs.
""" 
function transform(object::MbplsrRosa, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    nbl = length(object.xmeans)
    zX = list(nbl, Matrix{Float64})
    @inbounds for k = 1:nbl
        zX[k] = center(X[k], object.xmeans[k])
    end
    reduce(hcat, zX) * vcol(object.R, 1:nlv)
end

"""
    coef(object::MbplsrRosa; nlv = nothing)
Compute the X b-coefficients of a model fitted with `nlv` LVs.
* `object` : The maximal fitted model.
* `nlv` : Nb. LVs to consider. If nothing, it is the maximum nb. LVs.
""" 
function coef(object::MbplsrRosa; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    zxmeans = reduce(vcat, object.xmeans)
    beta = object.C[:, 1:nlv]'
    B = vcol(object.R, 1:nlv) * beta
    int = object.ymeans' .- zxmeans' * B
    (B = B, int = int)
end

"""
    predict(object::MbplsrRosa, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : A list (vector) of X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
    If nothing, it is the maximum nb. LVs.
""" 
function predict(object::MbplsrRosa, X; nlv = nothing)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    zX = reduce(hcat, X)
    pred = list(le_nlv, Matrix{Float64})
    @inbounds for k = 1:le_nlv
        z = coef(object; nlv = nlv[k])
        pred[k] = z.int .+ zX * z.B
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end




