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
* Same as function `pcasvd`

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
fitm = model.fitm ;
pnames(fitm)
pnames(fitm.fitm)

@head fitm.fitm.T
@head transf(model, X)

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
    par = recovkw(ParPca, kwargs).par
    Q = eltype(X)
    q = nco(Y)
    ymeans = colmean(Y, weights)
    yscales = ones(Q, q)  # built only for consistency with coef::Plsr
    fitm = pcasvd!(X, weights; kwargs...)
    ## Below, first term of the product is equal to Diagonal(1 ./ fitm.sv[1:nlv].^2) 
    ## if T is D-orthogonal. This is the case for the actual version (pcasvd)
    ## theta: coefs regression of Y on T (= C')
    ## not needed (same theta): fcenter!(Y, ymeans)
    theta = inv(fitm.T' * fweight(fitm.T, fitm.weights.w)) * fitm.T' * fweight(Y, fitm.weights.w)
    Pcr(fitm, theta', ymeans, yscales, par) 
end

""" 
    transf(object::Union{Pcr, Spcr}, X; nlv = nothing)
Compute latent variables (LVs = scores) from a fitted model and a matrix X.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Union{Pcr, Spcr}, X; nlv = nothing)
    transf(object.fitm, X; nlv)
end

"""
    coef(object::Pcr; nlv = nothing)
Compute the b-coefficients of a LV model.
* `object` : The fitted model.
* `nlv` : Nb. LVs to consider.

For a model fitted from X(n, p) and Y(n, q), the returned 
object `B` is a matrix (p, q). If `nlv` = 0, `B` is a matrix 
of zeros. The returned object `int` is the intercept.
""" 
function coef(object::Pcr; nlv = nothing)
    a = nco(object.fitm.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    theta = vcol(object.C, 1:nlv)'
    Dy = Diagonal(object.yscales)
    ## Not used for Spcr (since R not computed; while for Pcr, R = V)
    B = fweight(vcol(object.fitm.V, 1:nlv), 1 ./ object.fitm.xscales) * theta * Dy
    ## In 'int': No correction is needed, since 
    ## ymeans, xmeans and B are in the original scale 
    int = object.ymeans' .- object.fitm.xmeans' * B
    ## End
    (B = B, int = int)
end

"""
    predict(object::Pcr, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
function predict(object::Pcr, X; nlv = nothing)
    X = ensure_mat(X)
    a = nco(object.fitm.T)
    isnothing(nlv) ? nlv = a : nlv = (min(a, minimum(nlv)):min(a, maximum(nlv)))
    le_nlv = length(nlv)
    pred = list(Matrix{eltype(X)}, le_nlv)
    @inbounds for i in eachindex(nlv)
        coefs = coef(object; nlv = nlv[i])
        pred[i] = coefs.int .+ X * coefs.B  # try muladd(X, coefs.B, coefs.int)
    end 
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end

"""
    summary(object::Union{Pcr, Spcr}, X)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Union{Pcr, Spcr}, X)
    summary(object.fitm, X)
end




