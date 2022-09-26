struct Pcr
    fm_pca
    T::Matrix{Float64}
    R::Matrix{Float64}
    C::Matrix{Float64}
    xmeans::Vector{Float64}
    xscales::Vector{Float64}
    ymeans::Vector{Float64}
    yscales::Vector{Float64}
    weights::Vector{Float64}
end

"""
    pcr(X, Y, weights = ones(size(X, 1)); nlv,
        scal = false)
    pcr!(X::Matrix, Y::Matrix, weights = ones(size(X, 1)); nlv,
        scal = false)
Principal component regression (PCR) with a SVD factorization.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations.
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X`
    is scaled by its uncorrected standard deviation.

`weights` is internally normalized to sum to 1. 

`X` and `Y` are internally centered. 

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlv = 15
fm = pcr(Xtrain, ytrain; nlv = nlv) ;
pnames(fm)
fm.T

zcoef = coef(fm)
zcoef.int
zcoef.B
coef(fm; nlv = 7).B

fm_pca = fm.fm_pca ;
transform(fm_pca, Xtest)
transform(fm_pca, Xtest; nlv = 7)

res = predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
plotxy(vec(res.pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f   

res = predict(fm, Xtest; nlv = 1:2)
res.pred[1]
res.pred[2]

# See ?pcasvd
res = Base.summary(fm_pca, Xtrain)
res.explvarx
```
""" 
function pcr(X, Y, weights = ones(size(X, 1)); nlv, 
        scal = false)
    pcr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv, 
        scal = scal)
end

function pcr!(X::Matrix, Y::Matrix, weights = ones(size(X, 1)); nlv, 
        scal = false)
    q = nco(Y)
    weights = mweight(weights)
    ymeans = colmean(Y, weights)
    # No need to scale Y
    # Only for consistency with coef::Plsr  
    yscales = ones(q)
    # End 
    fm = pcasvd!(X, weights; nlv = nlv, scal = scal)
    D = Diagonal(fm.weights)
    beta = inv(fm.T' * D * fm.T) * fm.T' * D * Y
    # first term of the product = Diagonal(1 ./ fm.sv[1:nlv].^2) if T is D-orthogonal
    # This is the case for the actual version (pcasvd)
    Pcr(fm, fm.T, fm.P, beta', fm.xmeans, fm.xscales, ymeans, yscales, weights)
end






