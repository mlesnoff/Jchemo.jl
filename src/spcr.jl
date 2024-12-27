"""
    spcr(; kwargs...)
    spcr(X, Y; kwargs...)
    spcr(X, Y, weights::Weight; kwargs...)
    pcr!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Sparse principal component regression (sPCR). 
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
* `meth` : Method used for the sparse thresholding. 
    Possible values are: `:soft`, `:softs`, 
    `:hard`. See thereafter.
* `nvar` : Only used if `meth = :soft` or `meth = :hard`.
    Nb. variables (`X`-columns) selected for each principal
    component (PC). Can be a single integer (i.e. same nb. 
    of variables for each PC), or a vector of length `nlv`.   
* `delta` : Only used if `meth = :softs`. Constant used in function 
   `soft` for the thresholding on the loadings (after they are 
    standardized to their maximal absolute value). Must ∈ [0, 1].
    Higher is `delta`, stronger is the thresholding. 

Regression on scores computed with the sPCA-rSVD algorithm of Shen 
& Huang 2008 (regularized low rank matrix approximation). 

## References
Shen, H., Huang, J.Z., 2008. Sparse principal component 
analysis via regularized low rank matrix approximation. 
Journal of Multivariate Analysis 99, 1015–1034. 
https://doi.org/10.1016/j.jmva.2007.06.007

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlv = 15
meth = :soft ; nvar = 20
#meth = :hard ; nvar = 20
model = spcr(; nlv, meth, nvar) ;
fit!(model, Xtrain, ytrain)
pnames(model)
pnames(model.fitm)
@head model.fitm.T
@head model.fitm.P

@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

res = predict(model, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

res = summary(model, Xtrain) ;
pnames(res)
z = res.explvarx
plotgrid(z.nlv, z.cumpvar; step = 2, xlabel = "Nb. LVs", 
    ylabel = "Prop. Explained X-Variance").f
```
""" 
spcr(; kwargs...) = JchemoModel(spcr, nothing, kwargs)

function spcr(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    spcr(X, Y, weights; kwargs...)
end

function spcr(X, Y, weights::Weight; kwargs...)
    spcr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function spcr!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParSpcr, kwargs).par
    Q = eltype(X)
    q = nco(Y)
    ymeans = colmean(Y, weights)
    ## No need to fscale Y
    yscales = ones(Q, q)
    ## End 
    fitm = spca!(X, weights; kwargs...)
    K = fweight(fitm.T, weights.w)'
    beta = inv(K * fitm.T) * K * fcenter(Y, ymeans)
    Spcr(fitm, fitm.T, fitm.P, beta', fitm.xmeans, fitm.xscales, ymeans, yscales, weights,
        fitm.sellv, fitm.sel,  # add compared to ::Pcr
        par)
end

"""
    predict(object::Spcr, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Spcr, X; nlv = nothing)
    X = ensure_mat(X)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = (max(0, minimum(nlv)):min(a, maximum(nlv)))
    le_nlv = length(nlv)
    T = transf(object, X)
    pred = list(Matrix{eltype(X)}, le_nlv)
    @inbounds for i in eachindex(nlv)
        beta = vcol(object.C, 1:nlv[i])'
        pred[i] = vcol(T, 1:nlv[i]) * beta .+ object.ymeans'
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end



