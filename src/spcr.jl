"""
    spcr(; kwargs...)
    spcr(X, Y; kwargs...)
    spcr(X, Y, weights::ProbabilityWeights; kwargs...)
    spcr!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
Sparse principal component regression (sPCR). 
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. principal components (PCs).
* `meth` : Method used for the thresholding of the loadings. Possible values are: `:soft`, `:hard`. See thereafter.
* `defl` : Type of `X`-matrix deflation, see below.
* `nvar` : Nb. variables (`X`-columns) kept to build the PCs (non-zero loadings). Can be a single integer 
    (i.e. same nb. of variables for each PC), or a vector of length `nlv`.   
* `tol` : Tolerance value for stopping the Nipals iterations.
* `maxit` : Maximum nb. of Nipals iterations.
* `scal` : Boolean. If `true`, each column of `X` is scaled by its uncorrected standard deviation.

Regression (MLR) on scores computed from a sparse PCA (sPCA-rSVD algorithm of Shen & Huang 2008 ). 
See function `spca` for details.

## References
Shen, H., Huang, J.Z., 2008. Sparse principal component analysis via regularized low rank matrix approximation. 
Journal of Multivariate Analysis 99, 1015–1034. https://doi.org/10.1016/j.jmva.2007.06.007

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
meth = :soft
#meth = :hard
nvar = 10 
model = spcr(; nlv, meth, nvar, defl = :t) ;
fit!(model, Xtrain, ytrain)
@names model
fitm = model.fitm ;
@names fitm
typeof(fitm.fitm)
@names fitm.fitm

fitm.fitm.niter

fitm.fitm.sellv
fitm.fitm.sel

@head transf(model, Xtrain)
@head fitm.fitm.T

@head fitm.fitm.V

@head transf(model, Xtest)
@head transf(model, Xtest, 3)

res = predict(model, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

res = summary(model, Xtrain) ;
@names res
z = res.explvarx
plotgrid(z.nlv, z.cumpvar; step = 2, xlabel = "Nb. LVs", ylabel = "Prop. Explained X-Variance").f
```
""" 
spcr(; kwargs...) = JchemoModel(spcr, nothing, kwargs)

function spcr(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    n = nro(X)
    weights = pweight(ones(Q, n))
    spcr(X, Y, weights; kwargs...)
end

function spcr(X, Y, weights::ProbabilityWeights; kwargs...)
    spcr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function spcr!(X::Matrix{Q}, Y::Matrix{Q}, weights::ProbabilityWeights{Q}; kwargs...) where Q <: AbstractFloat
    par = recovkw(ParSpca, kwargs).par
    Q = eltype(X)
    q = nco(Y)
    ymeans = colmean(Y, weights)
    yscales = ones(Q, q) 
    fitm = spca!(X, weights; kwargs...)
    par.nlv = fitm.par.nlv
    theta = inv(fitm.T' * fweightr(fitm.T, fitm.weights.values)) * fitm.T' * fweightr(Y, fitm.weights.values)  # = C'
    Spcr(fitm, theta', ymeans, yscales, par) 
end

"""
    predict(object::Spcr, X)
    predict(object::Spcr, X, nlv::Union{Int, AbstractVector{Int}})
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
""" 
## Function 'coef' not yet implemented for Spcr, therefore 'predict' uses directly C (regression coefs on the scores) 
function predict(object::Spcr, X)
    T = transf(object, X)
    theta = object.C'
    pred = T * theta .+ object.ymeans'
    (pred = pred, nlv = object.par.nlv)
end

function predict(object::Spcr, X, nlv::Union{Int, AbstractVector{Int}})
    X = ensure_mat(X)
    Q = eltype(X)
    a = object.par.nlv
    if isa(nlv, Int)
        nlv = min(nlv, a)
    else
        nlv = min(minimum(nlv), a):min(maximum(nlv), a)
    end
    le_nlv = length(nlv)
    T = transf(object, X)
    pred = list(Matrix{Q}, le_nlv)
    @inbounds for i in eachindex(nlv)
        theta = vcol(object.C, 1:nlv[i])'
        pred[i] = vcol(T, 1:nlv[i]) * theta .+ object.ymeans'
    end 
    (pred = pred, nlv)
end


