"""
    rosaplsr(; kwargs...)
    rosaplsr(Xbl, Y; kwargs...)
    rosaplsr(Xbl, Y, weights::Weight; kwargs...)
    rosaplsr!(Xbl::Vector, Y::Matrix, weights::Weight; kwargs...)
Multiblock ROSA PLSR (Liland et al. 2016).
* `Xbl` : List of blocks (vector of matrices) of X-data 
    Typically, output of function `mblock` from (n, p) data.  
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs = scores T) to compute.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` 
    and `Y` is scaled by its uncorrected standard deviation 
    (before the block scaling).

The function has the following differences with the 
original algorithm of Liland et al. (2016):
* Scores T are not normed to 1.
* Multivariate `Y` is allowed. In such a case, 
    the squared residuals are summed over the columns 
    for finding the winning block for each global LV 
    (therefore Y-columns should have the same fscale).

## References
Liland, K.H., Næs, T., Indahl, U.G., 2016. ROSA — a fast 
extension of partial least squares regression for multiblock 
data analysis. Journal of Chemometrics 30, 651–662. 
https://doi.org/10.1002/cem.2824

## Examples
```julia
using JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "ham.jld2") 
@load db dat
pnames(dat) 
X = dat.X
Y = dat.Y
y = Y.c1
group = dat.group
listbl = [1:11, 12:19, 20:25]
s = 1:6
Xbltrain = mblock(X[s, :], listbl)
Xbltest = mblock(rmrow(X, s), listbl)
ytrain = y[s]
ytest = rmrow(y, s) 
ntrain = nro(ytrain) 
ntest = nro(ytest) 
ntot = ntrain + ntest 
(ntot = ntot, ntrain , ntest)

nlv = 3
scal = false
#scal = true
mo = rosaplsr(; nlv, scal)
fit!(mo, Xbltrain, ytrain)
pnames(mo) 
pnames(mo.fm)
@head mo.fm.T
@head transf(mo, Xbltrain)
transf(mo, Xbltest)

res = predict(mo, Xbltest)
res.pred 
rmsep(res.pred, ytest)
```
""" 
function rosaplsr(Xbl, Y; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    n = nro(Xbl[1])
    weights = mweight(ones(Q, n))
    rosaplsr(Xbl, Y, weights; kwargs...)
end

function rosaplsr(Xbl, Y, weights::Weight; kwargs...)
    Q = eltype(Xbl[1][1, 1])
    nbl = length(Xbl)  
    zXbl = list(Matrix{Q}, nbl)
    @inbounds for k = 1:nbl
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    rosaplsr!(zXbl, copy(ensure_mat(Y)), weights; kwargs...)
end

function rosaplsr!(Xbl::Vector, Y::Matrix, weights::Weight; kwargs...)
    par = recovkwargs(Par, kwargs)
    Q = eltype(Xbl[1][1, 1])   
    n = nro(Xbl[1])
    q = nco(Y)
    nlv = par.nlv
    nbl = length(Xbl)
    D = Diagonal(weights.w)
    fmsc = blockscal(Xbl, weights; bscal = :none, centr = true, 
        scal = par.scal)
    transf!(fmsc, Xbl)
    ymeans = colmean(Y, weights)
    yscales = ones(Q, q)
    if par.scal 
        yscales .= colstd(Y, weights)
        fcscale!(Y, ymeans, yscales)
    else
        fcenter!(Y, ymeans)
    end
    p = [nco(Xbl[k]) for k = 1:nbl]
    ## Pre-allocation
    W = similar(Xbl[1], sum(p), nlv)
    P = copy(W)
    T = similar(Xbl[1], n, nlv)
    TT = similar(Xbl[1], nlv)    
    C = similar(Xbl[1], q, nlv)
    DY = similar(Xbl[1], n, q)
    t   = similar(Xbl[1], n)
    dt  = similar(Xbl[1], n)   
    c   = similar(Xbl[1], q)
    zp_bl = list(Vector{Q}, nbl)
    zp = similar(Xbl[1], sum(p))
    #ssr = similar(Xbl[1], nbl)
    corr = similar(Xbl[1], nbl)
    Wbl = list(Array, nbl)
    wbl = list(Vector{Q}, nbl)      # List of the weights "w" by block for a given "a"
    zT = similar(Xbl[1], n, nbl)    # Matrix gathering the nbl scores for a given "a"
    bl = fill(0, nlv)
    #Res = zeros(n, q, nbl)
    ## Start 
    @inbounds for a = 1:nlv
        DY .= D * Y  # apply the metric on covariance
        @inbounds for k = 1:nbl
            XtY = Xbl[k]' * DY
            if q == 1
                wbl[k] = vec(XtY)
                #wbl[k] = vec(cor(Xbl[k], Y))
                wbl[k] ./= norm(wbl[k])
            else
                wbl[k] = svd!(XtY).U[:, 1]
            end
            zT[:, k] .= Xbl[k] * wbl[k]
        end
        ## GS Orthogonalization of the scores
        if a > 1
            z = vcol(T, 1:(a - 1))
            zT .= zT .- z * inv(z' * (D * z)) * z' * (D * zT)
        end
        ## Selection of the winner block (opt)
        @inbounds for k = 1:nbl
            t = vcol(zT, k)
            corr[k] = sum(corm(Y, t, weights).^2)
        end
        opt = argmax(corr)
        ## Faster than:
        ## Old
        #@inbounds for k = 1:nbl
        #    t = vcol(zT, k)
        #    dt .= weights.w .* t
        #    tt = dot(t, dt)
        #    Res[:, :, k] .= Y .- (t * t') * DY / tt
        #end
        #ssr = vec(sum(Res.^2, dims = (1, 2)))
        #opt = findmin(ssr)[2][1]
        ## End
        bl[a] = opt
        ## Outputs for winner
        t .= zT[:, opt]
        dt .= weights.w .* t
        tt = dot(t, dt)
        mul!(c, Y', dt)
        c ./= tt     
        T[:, a] .= t
        TT[a] = tt
        C[:, a] .= c
        ## Old
        #Y .= Res[:, :, opt]
        ## End
        Y .-= (t * t') * DY / tt
        for k = 1:nbl
            zp_bl[k] = Xbl[k]' * dt
        end
        zp .= reduce(vcat, zp_bl)
        P[:, a] .= zp / tt
        ## Orthogonalization of the weights "w" 
        ## by block
        zw = wbl[opt]
        if (a > 1) && isassigned(Wbl, opt)       
            zW = Wbl[opt]
            zw .= zw .- zW * (zW' * zw)
        end
        zw ./= norm(zw)
        if !isassigned(Wbl, opt) 
            Wbl[opt] = reshape(zw, :, 1)
        else
            Wbl[opt] = hcat(Wbl[opt], zw)
        end
        ## Build the weights over the overall 
        ## matrix
        z = zeros(Q, nbl) ; z[opt] = 1
        W[:, a] .= reduce(vcat, z .* wbl)
    end
    R = W * inv(P' * W)
    Rosaplsr(T, P, R, W, C, TT, fmsc, 
        ymeans, yscales, weights, bl, kwargs, par)
end

""" 
    transf(object::Rosaplsr, Xbl; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which LVs are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transf(object::Rosaplsr, Xbl; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    zXbl = transf(object.fmsc, Xbl)
    reduce(hcat, zXbl) * vcol(object.R, 1:nlv)
end

"""
    coef(object::Rosaplsr; nlv = nothing)
Compute the X b-coefficients of a model fitted with `nlv` LVs.
* `object` : The fitted model.
* `nlv` : Nb. LVs to consider.
""" 
function coef(object::Rosaplsr; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    zxmeans = reduce(vcat, object.fmsc.xmeans)
    beta = object.C[:, 1:nlv]'
    xscales = reduce(vcat, object.fmsc.xscales)
    W = Diagonal(object.yscales)
    B = Diagonal(1 ./ xscales) * vcol(object.R, 1:nlv) * beta * W
    int = object.ymeans' .- zxmeans' * B
    (B = B, int = int)
end

"""
    predict(object::Rosaplsr, Xbl; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Rosaplsr, Xbl; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    Q = eltype(Xbl[1][1, 1])
    X = reduce(hcat, Xbl)
    pred = list(Matrix{Q}, le_nlv)
    @inbounds for i = 1:le_nlv
        z = coef(object; nlv = nlv[i])
        pred[i] = z.int .+ X * z.B
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end




