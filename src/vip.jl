"""
    vip(object::Union{Pcr, Plsr}; nlv = nothing)
    vip(object::Union{Pcr, Plsr}, Y; nlv = nothing)
Variable importance on PLS projections (VIP).
* `object` : The fitted model (object of structure `Plsr`).
* `Y` : The Y-data that was used to fit the model.
* `nlv` : Nb. latent variables (LVs) to consider.

For a PLS model (X, Y) with a number of A latent variables, 
and variable xj (column j of X): 
* VIP(xj) = Sum(a=1,...,A) R2(Yc, ta) waj^2 / Sum(a=1,...,A) R2(Yc, ta) (1 / p) 
where:
* Yc is the centered Y, 
* ta is the ath X-score, 
* and R2(Yc, ta) the proportion of Yc-variance explained by ta, 
    i.e. ||Yc.hat||^2 / ||Yc||^2 (where Yc.hat is the LS estimate of Yc by ta).  

When `Y` is used as argument, R2(Yc, ta) is replaced by the redundancy
Rd(Yc, ta) (see function `rd`), such as in Tenenhaus 1998 p.139. 

The function also works for PCR models.

## References
Chong, I.-G., Jun, C.-H., 2005. Performance of some variable selection methods when 
multicollinearity is present. Chemometrics and Intelligent Laboratory Systems 78, 103–112. 
https://doi.org/10.1016/j.chemolab.2004.12.011

Mehmood, T., Sæbø, S., Liland, K.H., 2020. Comparison of variable selection methods 
in partial least squares regression. Journal of Chemometrics 34, e3226. 
https://doi.org/10.1002/cem.3226

Tenenhaus, M., 1998. La régression PLS: théorie et pratique. 
Editions Technip, Paris.

## Examples
```julia
X = [1. 2 3 4; 4 1 6 7; 12 5 6 13; 27 18 7 6; 12 11 28 7] 
Y = [10. 11 13; 120 131 27; 8 12 4; 1 200 8; 100 10 89] 
y = Y[:, 1] 
ycla = [1; 1; 1; 2; 2]

nlv = 3
fm = plskern(X, Y; nlv = nlv) ;
res = vip(fm)
pnames(res)
mean(res.imp.^2)
vip(fm; nlv = 1).imp

nlv = 2
fm = plsrda(X, ycla; nlv = nlv) ;
fmpls = fm.fm
vip(fmpls).imp
Ydummy = dummy(ycla).Y
vip(fmpls, Ydummy).imp

nlv = 2
fm = plslda(X, ycla; nlv = nlv) ;
fmpls = fm.fm.fm_pls
vip(fmpls).imp
Ydummy = dummy(ycla).Y
vip(fmpls, Ydummy).imp
```
""" 
function vip(object::Union{Pcr, Plsr}; nlv = nothing)
    if isa(object, Jchemo.Pcr)
        W = object.fm_pca.P
    else
        W = object.W
    end
    a = nco(object.T)
    p = nro(W)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    sqrtw = sqrt.(object.weights)
    ## ::Plsr represents algorithmns where W is normed
    ## ==> No need to do the fllowing: 
    ## wnorms = colnorm(W)
    ## W2 = scale(W, wnorms).^2
    ## End
    W2 = W[:, 1:nlv].^2
    sst = zeros(nlv)
    for a = 1:nlv
        t = sqrtw .* object.T[:, a]
        tt = dot(t, t)
        beta = object.C[:, a]'
        sst[a] = tr(beta' * beta * tt)
    end 
    A = rowsum(sst' .* W2)
    B = sum(sst) * (1 / p)
    imp = sqrt.(A / B)
    (imp = imp, W2, sst)
end

function vip(object::Union{Pcr, Plsr}, Y; nlv = nothing)
    if isa(object, Jchemo.Pcr)
        W = object.fm_pca.P
    else
        W = object.W
    end
    a = nco(object.T)
    p = nro(W)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    W2 = W[:, 1:nlv].^2
    rdd = rd(Y, object.T[:, 1:nlv], object.weights)
    A = rowsum(rdd .* W2)
    B = sum(rdd) * (1 / p)
    imp = sqrt.(A / B)
    (imp = imp, W2, rdd)
end





