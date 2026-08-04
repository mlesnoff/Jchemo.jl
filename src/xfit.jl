"""
    xfit(object, X, nlv::Int = nco(object.T))
    xfit!(object, X::Matrix{Q}, nlv::Int = nco(object.T)) where Q <: Float
Fit a matrix from a bilinear model (e.g., PCA).
* `object` : The reduction dimension model that was fitted on the data (e.g., object `fitm` returned by functions 
    `pcasvd` or `plskern`).
* `X` : X-data to be approximated from the model. Must be in the same scale as the X-data used to fit
    model `object`, i.e. before centering and eventual scaling.
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to consider. By default, it is the maximum nb. of LVs
    defined in model `object`.

Compute an approximate of matrix `X` from a bilinear model (e.g., PCA or PLS) fitted on `X`. The computed approximate X 
is returned in the original location and scale of the X-data used to fit model `object`.

## Examples 
```julia 
using Jchemo

X = [1. 2 3 4; 4 1 6 7; 12 5 6 13; 
    27 18 7 6; 12 11 28 7] 
Y = [10. 11 13; 120 131 27; 8 12 4; 
    1 200 8; 100 10 89] 
n, p = size(X)
Xnew = X[1:3, :]
Ynew = Y[1:3, :]
y = Y[:, 1]
ynew = Ynew[:, 1]
weights = pweight(rand(n))

#### Pca

nlv = 2 
scal = :none
#scal = :std
model = pcasvd(; nlv, scal) ;
fit!(model, X)
fitm = model.fitm ;
@head xfit(fitm, X)
@head xfit(fitm, X, 1)
@head xfit(fitm, X, 0)
fitm.xmeans
xfit(fitm, Xnew)
xfit(fitm, Xnew, 1)

@head X
@head xfit(fitm, X) + xresid(fitm, X)
@head xfit(fitm, X, 1) + xresid(fitm, X, 1)

@head Xnew
@head xfit(fitm, Xnew) + xresid(fitm, Xnew)

model = pcasvd(; nlv = min(n, p), scal) 
fit!(model, X)
fitm = model.fitm ;
@head xfit(fitm, X)
@head xresid(fitm, X)

#### Pls

nlv = 3
scal = :none
#scal = :std
model = plskern(; nlv, scal)
fit!(model, X, Y, weights) 
fitm = model.fitm ;
@head xfit(fitm, X)
@head xfit(fitm, X, 1)
@head xfit(fitm, X, 0)
colmean(X, weights)
xfit(fitm, Xnew)
@head xfit(fitm, Xnew, 1)

@head X
@head xfit(fitm, X) + xresid(fitm, X)
@head xfit(fitm, X, 1) + xresid(fitm, X, 1)

@head Xnew
@head xfit(fitm, Xnew) + xresid(fitm, Xnew)

model = plskern(; nlv = min(n, p), scal) 
fit!(model, X, Y, weights) 
fitm = model.fitm ;
@head xfit(fitm, X) 
@head xresid(fitm, X) 
xfit(fitm, Xnew)
xresid(fitm, Xnew)
```
""" 
function xfit(object, X, nlv::Int = nco(object.T))  
    xfit!(object, copy(ensure_mat(X)), nlv)
end

function xfit!(object, X::Matrix{Q}, nlv::Int = nco(object.T)) where Q <: Float
    nlv = min(nlv, nco(object.T))
    if nlv == 0
        @inbounds for i in axes(X, 1)
            X[i, :] .= object.xmeans
        end
    else
        T = transf(object, X, nlv)
        mul!(X, T, vcol(object.V, 1:nlv)')
        ## Coming back to the original scale
        fscale!(X, 1 ./ object.xscales)    
        fcenter!(X, -object.xmeans)
    end
    X
end
