"""
    pcr(; kwargs...)
    pcr(X, Y; kwargs...)
    pcr(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
    pcr!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
Principal component regression (PCR) with a SVD factorization.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. of principal components (PCs).
* `scal` : Symbol defining the column scaling of `X`. Possible values are: `:none`, `std` (uncorrected STD), 
    `prt` (pareto) and `:mad` (MAD).

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat
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
@names model
fitm = model.fitm ;
@names fitm
typeof(fitm.fitm)
@names fitm.fitm

@head transf(model, Xtrain)
@head fitm.fitm.T

@head transf(model, Xtest)
@head transf(model, Xtest, 3)

coef(model)
coef(model, 3)

res = predict(model, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

res = predict(model, Xtest, 1:2)
@head res.pred[1]
@head res.pred[2]

res = summary(model, Xtrain) ;
@names res
z = res.explvarx
plotgrid(z.nlv, z.cumpvar; step = 2, xlabel = "Nb. LVs", ylabel = "Prop. Explained X-Variance").f
```
""" 
pcr(; kwargs...) = JchemoModel(pcr, nothing, kwargs)

function pcr(X, Y; kwargs...)
    X = ensure_mat(X)
    weights = pweight(ones(eltype(X), nro(X)))
    pcr(X, Y, weights; kwargs...)
end

function pcr(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
    pcr!(copy(X), copy(Y), weights; kwargs...)
end

function pcr!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
    par = recovkw(ParPca, kwargs).par
    q = nco(Y)
    ymeans = colmean(Y, weights)
    yscales = ones(Q, q)  # built only for consistency with coef::Plsr
    fitm = pcasvd!(X, weights; kwargs...)
    par.nlv = fitm.par.nlv
    ## Below, first term of the product is equal to Diagonal(1 ./ fitm.sv[1:nlv].^2) 
    ## if T is D-orthogonal. This is the case for the actual version (pcasvd)
    ## theta: coefs regression of Y on T (= C')
    ## not needed (same theta): fcenter!(Y, ymeans)
    theta = inv(fitm.T' * fweightr(fitm.T, fitm.weights.values)) * fitm.T' * fweightr(Y, fitm.weights.values)  # = C'
    Pcr(fitm, theta', ymeans, yscales, par) 
end

""" 
    transf(object::Union{Pcr, Spcr}, X)
    transf(object::Union{Pcr, Spcr}, X, nlv::Int)
Compute latent variables (LVs; = scores) from a fitted model and a matrix X.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
transf(object::Union{Pcr, Spcr}, X) = transf(object.fitm, X)

transf(object::Union{Pcr, Spcr}, X, nlv::Int) = transf(object.fitm, X, nlv)

"""
    coef(object::Pcr)
    coef(object::Pcr, nlv::Int)
Compute the b-coefficients of a LV model.
* `object` : The fitted model.
* `nlv` : Nb. LVs to consider.

For a model fitted from X (n, p) and Y (n, q), the returned object `B` is a matrix (p, q). If `nlv` = 0, `B` is a matrix 
of zeros. The returned object `int` is the intercept.
""" 
function coef(object::Pcr)
    theta = object.C'
    Dy = Diagonal(object.yscales)
    ## Not used for Spcr (since R not computed; while for Pcr, R = V)
    B = fweightr(object.fitm.V, 1 ./ object.fitm.xscales) * theta * Dy
    ## In 'int': No correction is needed, since ymeans, xmeans and B are in the original scale 
    int = object.ymeans' .- object.fitm.xmeans' * B
    ## End
    (B = B, int, nlv = object.par.nlv)
end

function coef(object::Pcr, nlv::Int)
    a = object.par.nlv
    nlv = isnothing(nlv) ? a : min(nlv, a)
    theta = vcol(object.C, 1:nlv)'
    Dy = Diagonal(object.yscales)
    ## Not used for Spcr (since R not computed; while for Pcr, R = V)
    B = fweightr(vcol(object.fitm.V, 1:nlv), 1 ./ object.fitm.xscales) * theta * Dy
    ## In 'int': No correction is needed, since ymeans, xmeans and B are in the original scale 
    int = object.ymeans' .- object.fitm.xmeans' * B
    ## End
    (B = B, int, nlv)
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




