"""
    rosaplsr(; kwargs...)
    rosaplsr(Xbl, Y; kwargs...)
    rosaplsr(Xbl, Y, weights::Weight; kwargs...)
    rosaplsr!(Xbl::Vector, Y::Matrix, weights::Weight; kwargs...)
Multiblock ROSA PLSR (Liland et al. 2016).
* `Xbl` : List of blocks (vector of matrices) of X-data. Typically, output of function `mblock` 
    from (n, p) data.  
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs = scores) to compute.
* `scal` : Boolean. If `true`, each column of blocks in `Xbl` and `Y` is scaled by its uncorrected 
    standard deviation (before the block scaling).

The function has the following differences with the original algorithm of Liland et al. (2016):
* Scores T (latent variables LVs) are not normed to 1.
* Multivariate `Y` is allowed. In such a case, the squared residuals are summed over the columns 
    to find the winning block for each global LV (therefore, Y-columns should have the same scale).

## References
Liland, K.H., Næs, T., Indahl, U.G., 2016. ROSA — a fast extension of partial least squares regression 
for multiblock data analysis. Journal of Chemometrics 30, 651–662. https://doi.org/10.1002/cem.2824

## Examples
```julia
using Jchemo, JchemoData, JLD2
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "ham.jld2") 
@load db dat
@names dat 
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
model = rosaplsr(; nlv, scal)
fit!(model, Xbltrain, ytrain)
@names model 
@names model.fitm
@head model.fitm.T
@head transf(model, Xbltrain)
transf(model, Xbltest)

res = predict(model, Xbltest)
res.pred 
rmsep(res.pred, ytest)
```
""" 
rosaplsr(; kwargs...) = JchemoModel(rosaplsr, nothing, kwargs)

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
    @inbounds for k in eachindex(Xbl)
        zXbl[k] = copy(ensure_mat(Xbl[k]))
    end
    rosaplsr!(zXbl, copy(ensure_mat(Y)), weights; kwargs...)
end

function rosaplsr!(Xbl::Vector, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParSoplsr, kwargs).par
    Q = eltype(Xbl[1][1, 1])   
    n = nro(Xbl[1])
    pbl = [nco(Xbl[k]) for k in eachindex(Xbl)]
    q = nco(Y)
    nlv = par.nlv
    nbl = length(Xbl)
    fitmbl = blockscal(Xbl, weights; bscal = :none, centr = true, scal = par.scal)
    transf!(fitmbl, Xbl)
    ymeans = colmean(Y, weights)
    yscales = ones(Q, q)
    if par.scal 
        yscales .= colstd(Y, weights)
        fcscale!(Y, ymeans, yscales)
    else
        fcenter!(Y, ymeans)
    end
    ## Pre-allocation
    W = similar(Xbl[1], sum(pbl), nlv)
    V = copy(W)
    T = similar(Xbl[1], n, nlv)
    TT = similar(Xbl[1], nlv)    
    C = similar(Xbl[1], q, nlv)
    DY = similar(Xbl[1], n, q)
    t   = similar(Xbl[1], n)
    dt  = similar(Xbl[1], n)   
    c   = similar(Xbl[1], q)
    vbl = list(Vector{Q}, nbl)
    v = similar(Xbl[1], sum(pbl))
    corr = similar(Xbl[1], nbl)
    Wbl = list(Array{Q}, nbl)
    wbl = list(Vector{Q}, nbl)      # List of the weights "w" by block for a given "a"
    zT = similar(Xbl[1], n, nbl)    # Matrix gathering the nbl scores for a given "a"
    bl = fill(0, nlv)
    ## Old
    #ssr = similar(Xbl[1], nbl)
    #Res = zeros(n, q, nbl)
    ## Start 
    @inbounds for a = 1:nlv
        DY .= fweight(Y, weights.w)  # apply the metric to the covariance
        @inbounds for k in eachindex(Xbl)
            XtY = Xbl[k]' * DY
            if q == 1
                wbl[k] = vec(XtY)
                #wbl[k] = vec(cor(Xbl[k], Y))
                wbl[k] ./= normv(wbl[k])
            else
                wbl[k] = svd!(XtY).U[:, 1]
            end
            zT[:, k] .= Xbl[k] * wbl[k]
        end
        ## GS Orthogonalization of the scores
        if a > 1
            z = vcol(T, 1:(a - 1))
            zT .= zT .- z * inv(z' * fweight(z, weights.w)) * z' * fweight(zT, weights.w)
        end
        ## Selection of the winner block (opt)
        @inbounds for k in eachindex(Xbl)
            t = vcol(zT, k)
            corr[k] = sum(corm(Y, t, weights).^2)
        end
        opt = argmax(corr)
        ## Faster than:
        ## Old
        #@inbounds for k in eachindex(Xbl)
        #    t = vcol(zT, k)
        #    dt .= weights.w .* t
        #    tt = dot(t, dt)
        #    Res[:, :, k] .= Y .- (t * t') * DY / tt
        #end
        #ssr = vec(sum(Res.^2, dims = (1, 2)))
        #opt = findmin(ssr)[2][1]
        ## End
        bl[a] = opt
        ## Outputs for winner block
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
        for k in eachindex(Xbl)
            vbl[k] = Xbl[k]' * dt
        end
        v .= reduce(vcat, vbl)
        V[:, a] .= v / tt
        ## Orthogonalization of the weights "w" by block
        zw = wbl[opt]
        if (a > 1) && isassigned(Wbl, opt)       
            zW = Wbl[opt]
            zw .= zw .- zW * (zW' * zw)
        end
        zw ./= normv(zw)
        if !isassigned(Wbl, opt) 
            Wbl[opt] = reshape(zw, :, 1)
        else
            Wbl[opt] = hcat(Wbl[opt], zw)
        end
        ## Build the weights over the overall matrix
        z = zeros(Q, nbl)
        z[opt] = 1
        W[:, a] .= reduce(vcat, z .* wbl)
    end
    R = W * inv(V' * W)
    Rosaplsr(T, V, R, W, C, TT, fitmbl, ymeans, yscales, weights, bl, par)
end

""" 
    transf(object::Rosaplsr, Xbl; nlv = nothing)
Compute latent variables (LVs = scores) from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) 
    of X-data for which LVs are computed.
* `nlv` : Nb. LVs to compute.
""" 
function transf(object::Rosaplsr, Xbl; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    zXbl = transf(object.fitmbl, Xbl)
    fconcat(zXbl) * vcol(object.R, 1:nlv)
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
    xmeans = reduce(vcat, object.fitmbl.xmeans)
    xscales = reduce(vcat, object.fitmbl.xscales)
    theta = vcol(object.C, 1:nlv)'  # coefs regression of Y on T
    Dy = Diagonal(object.yscales)
    B = fweight(vcol(object.R, 1:nlv), 1 ./ xscales) * theta * Dy
    int = object.ymeans' .- xmeans' * B
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
    isnothing(nlv) ? nlv = a : nlv = min(a, minimum(nlv)):min(a, maximum(nlv))
    le_nlv = length(nlv)
    Q = eltype(Xbl[1][1, 1])
    X = reduce(hcat, Xbl)
    pred = list(Matrix{Q}, le_nlv)
    @inbounds for i in eachindex(nlv)
        z = coef(object; nlv = nlv[i])
        pred[i] = z.int .+ X * z.B
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end

