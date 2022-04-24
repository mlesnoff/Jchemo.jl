"""
    pcr(X, Y, weights = ones(size(X, 1)); nlv)
    pcr!(X::Matrix, Y::Matrix, weights = ones(size(X, 1)); nlv)
Principal component regression (PCR) with a SVD factorization.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations.
* `nlv` : Nb. latent variables (LVs) to compute.

`weights` is internally normalized to sum to 1. 

`X` and `Y` are internally centered. 

## Examples

```julia
using JLD2, CairoMakie
mypath = joinpath(@__DIR__, "..", "data")
db = string(mypath, "\\", "cassav.jld2") 
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
f, ax = scatter(vec(res.pred), ytest)
abline!(ax, 0, 1)
f

res = predict(fm, Xtest; nlv = 1:2)
res.pred[1]
res.pred[2]

# See ?pcasvd
res = Base.summary(fm_pca, Xtrain)
res.explvar
```
""" 
function pcr(X, Y, weights = ones(size(X, 1)); nlv)
    pcr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; nlv = nlv)
end

function pcr!(X::Matrix, Y::Matrix, weights = ones(size(X, 1)); nlv)
    weights = mweight(weights)
    ymeans = colmean(Y, weights)  
    center!(Y, ymeans)
    fm = pcasvd!(X, weights; nlv = nlv)
    D = Diagonal(fm.weights)
    beta = inv(fm.T' * D * fm.T) * fm.T' * D * Y
    # first term = Diagonal(1 ./ fm.sv[1:nlv].^2) if T is D-orthogonal
    # This is the case for the actual version (pcasvd)
    Pcr(fm, fm.T, fm.P, beta', fm.xmeans, ymeans, weights)
end






