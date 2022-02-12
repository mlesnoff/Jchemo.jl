"""
    pcr(X, Y, weights = ones(size(X, 1)); nlv)
Principal component regression (PCR) with a SVD factorization.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `nlv` : Nb. latent variables (LVs) to compute.

`X` and `Y` are internally centered. 
The model is computed with an intercept.

## Examples

```julia
n = 10 ; p = 5 ; q = 2
X = rand(n, p)
Y = rand(n, q)

nlv = 3
fm = pcr(X, Y, w; nlv = nlv) ;
Jchemo.coef(fm).B
Jchemo.coef(fm).int
Jchemo.coef(pcr(X, Y[:, 2], w; nlv = nlv)).B
Jchemo.coef(fm; nlv = 1).B

nlv = 3
fm = pcr(X, Y, w; nlv = nlv) ;
Jchemo.predict(fm, X[1:3, :]).pred
Jchemo.predict(plskern(X, Y, w; nlv = nlv), X[1:3, :]).pred

zfm = fm.fm_pca ;
res = summary(zfm, X) ;
pnames(res)
res.explvar

fm.T
Jchemo.transform(zfm, X[1:3, :])
```
""" 
function pcr(X, Y, weights = ones(size(X, 1)); nlv)
    pcr!(copy(X), copy(Y), weights; nlv = nlv)
end

function pcr!(X, Y, weights = ones(size(X, 1)); nlv)
    Y = ensure_mat(Y)
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






