"""
    pcr(X, Y, weights = ones(nro(X)); nlv,
        scal::Bool = false)
    pcr!(X::Matrix, Y::Matrix, weights = ones(nro(X)); nlv,
        scal::Bool = false)
Principal component regression (PCR) with a SVD factorization.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Internally normalized to sum to 1.
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X`
    is scaled by its uncorrected standard deviation.

`X` and `Y` are internally centered. 

## Examples
```julia
using JchemoData, JLD2, CairoMakie
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
fm = pcr(Xtrain, ytrain; nlv = nlv) ;
pnames(fm)
fm.T

zcoef = Jchemo.coef(fm)
zcoef.int
zcoef.B
Jchemo.coef(fm; nlv = 7).B

fmpca = fm.fmpca ;
transf(fmpca, Xtest)
transf(fmpca, Xtest; nlv = 7)

res = Jchemo.predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f   

res = Jchemo.predict(fm, Xtest; nlv = 1:2)
res.pred[1]
res.pred[2]

# See ?pcasvd
res = Base.summary(fmpca, Xtrain)
res.explvarx
```
""" 
function pcr(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    pcr(X, Y, weights; values(kwargs)...)
end

function pcr(X, Y, weights::Weight; kwargs...)
    pcr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), 
        weights; values(kwargs)...)
end

function pcr!(X::Matrix, Y::Matrix, weights::Weight; 
        kwargs...)
    Q = eltype(X)
    q = nco(Y)
    ymeans = colmean(Y, weights)
    ## No need to fscale Y
    ## below yscales is built only for consistency with coef::Plsr  
    yscales = ones(Q, q)
    # End 
    fm = pcasvd!(X, weights; kwargs...)
    D = Diagonal(fm.weights.w)
    ## Below, first term of the product = Diagonal(1 ./ fm.sv[1:nlv].^2) if T is D-orthogonal
    ## This is the case for the actual version (pcasvd)
    beta = inv(fm.T' * D * fm.T) * fm.T' * D * Y
    Pcr(fm, fm.T, fm.P, beta', fm.xmeans, fm.xscales, ymeans, yscales, weights)
end

""" 
    transf(object::Pcr, X; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and a matrix X.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Pcr, X; nlv = nothing)
    transf(object.fmpca, X; nlv = nlv)
end

"""
    summary(object::Pcr, X)
Summarize the fitted model.
* `object` : The fitted model.
* `X` : The X-data that was used to fit the model.
""" 
function Base.summary(object::Pcr, X)
    summary(object.fmpca, X)
end




