"""
    xfit(object)
    xfit(object, X; nlv = nothing)
    xfit!(object, X::Matrix; nlv = nothing)
Matrix fitting from a bilinear model (e.g. PCA).
* `object` : The fitted model.
* `X` : New X-data to be approximated from the model.
    Must be in the same scale as the X-data used to fit
    the model `object`, i.e. before centering 
    and eventual scaling.
Keyword arguments:
* `nlv` : Nb. components (PCs or LVs) to consider. 
    If `nothing`, it is the maximum nb. of components.

Compute an approximate of matrix `X` from a bilinear 
model (e.g. PCA or PLS) fitted on `X`. The fitted X is 
returned in the original scale of the X-data used to fit 
the model `object`.

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
weights = mweight(rand(n))

nlv = 2 
scal = false
#scal = true
model = pcasvd; nlv, scal) ;
fit!(model, X)
fm = model.fm ;
@head xfit(fm)
xfit(fm, Xnew)
xfit(fm, Xnew; nlv = 0)
xfit(fm, Xnew; nlv = 1)
fm.xmeans

@head X
@head xfit(fm) + xresid(fm, X)
@head xfit(fm, X; nlv = 1) + xresid(fm, X; nlv = 1)

@head Xnew
@head xfit(fm, Xnew) + xresid(fm, Xnew)

model = pcasvd; nlv = min(n, p), scal) 
fit!(model, X)
fm = model.fm ;
@head xfit(fm) 
@head xfit(fm, X)
@head xresid(fm, X)

nlv = 3
scal = false
#scal = true
model = plskern; nlv, scal)
fit!(model, X, Y, weights) 
fm = model.fm ;
@head xfit(fm)
xfit(fm, Xnew)
xfit(fm, Xnew, nlv = 0)
xfit(fm, Xnew, nlv = 1)

@head X
@head xfit(fm) + xresid(fm, X)
@head xfit(fm, X; nlv = 1) + xresid(fm, X; nlv = 1)

@head Xnew
@head xfit(fm, Xnew) + xresid(fm, Xnew)

model = plskern; nlv = min(n, p), scal) 
fit!(model, X, Y, weights) 
fm = model.fm ;
@head xfit(fm) 
@head xfit(fm, Xnew)
@head xresid(fm, Xnew)
```
""" 
function xfit(object)
    X = object.T * object.P'
    ## Coming back to the original scale
    fscale!(X, 1 ./ object.xscales)    
    fcenter!(X, -object.xmeans)
    X
end

function xfit(object, X; nlv = nothing)
    xfit!(object, copy(ensure_mat(X)); nlv)
end

function xfit!(object, X::Matrix; nlv = nothing)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    if nlv == 0
        m = nro(X)
        @inbounds for i = 1:m
            X[i, :] .= object.xmeans
        end
    else
        P = vcol(object.P, 1:nlv)
        mul!(X, transf(object, X; nlv), P')
        ## Coming back to the original scale
        fscale!(X, 1 ./ object.xscales)    
        fcenter!(X, -object.xmeans)
    end
    X
end
