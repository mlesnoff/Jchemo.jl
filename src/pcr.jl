"""
    pcr(; kwargs...)
    pcr(X, Y; kwargs...)
    pcr(X, Y, weights::Weight; kwargs...)
    pcr!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
Principal component regression (PCR) with a SVD factorization.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

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
model = pcr(; nlv) ;
fit!(model, Xtrain, ytrain)
pnames(model)
pnames(model.fitm)
@head model.fitm.T

coef(model)
coef(model; nlv = 3)

@head transf(model, Xtest)
@head transf(model, Xtest; nlv = 3)

res = predict(model, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",  
    ylabel = "Observed").f    

res = predict(model, Xtest; nlv = 1:2)
@head res.pred[1]
@head res.pred[2]

res = summary(model, Xtrain) ;
pnames(res)
z = res.explvarx
plotgrid(z.nlv, z.cumpvar; step = 2, xlabel = "Nb. LVs", ylabel = "Prop. Explained X-Variance").f
```
""" 
pcr(; kwargs...) = JchemoModel(pcr, nothing, kwargs)

function pcr(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    pcr(X, Y, weights; kwargs...)
end

function pcr(X, Y, weights::Weight; kwargs...)
    pcr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function pcr!(X::Matrix, Y::Matrix, weights::Weight; kwargs...)
    par = recovkw(ParPcr, kwargs).par
    Q = eltype(X)
    q = nco(Y)
    ymeans = colmean(Y, weights)
    ## No need to fscale Y
    ## below yscales is built only for consistency with coef::Plsr  
    yscales = ones(Q, q)
    ## End 
    fitm = pcasvd!(X, weights; kwargs...)
    D = Diagonal(fitm.weights.w)
    ## Below, first term of the product = Diagonal(1 ./ fitm.sv[1:nlv].^2) if T is D-orthogonal
    ## This is the case for the actual version (pcasvd)
    K = fitm.T' * D 
    beta = inv(K * fitm.T) * K * Y
    Pcr(fitm, fitm.T, fitm.P, beta', fitm.xmeans, fitm.xscales, ymeans, yscales, weights, par)
end

""" 
    transf(object::Union{Pcr, Spcr}, X; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and a matrix X.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Union{Pcr, Spcr}, X; nlv = nothing)
    transf(object.fitmpca, X; nlv)
end

"""
    summary(object::Union{Pcr, Spcr}, X)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Union{Pcr, Spcr}, X)
    summary(object.fitmpca, X)
end




