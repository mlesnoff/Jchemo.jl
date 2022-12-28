"""
    vip(object::Plsr; nlv = nothing)
    vip(object::Plsr, Y; nlv = nothing)
Variable importance on PLS projections (VIP).
* `object` : The fitted model (object of structure `Plsr`).
* `Y` : The Y-data that was used to fit the model.
* `nlv` : Nb. latent variables (LVs) to consider.
    



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


```
""" 




function vip(object::Plsr; nlv = nothing)
    a = nco(object.T)
    p = nro(object.W)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    sqrtw = sqrt.(object.weights)
    #wnorms = colnorm(object.W, object.weights)
    #W2 = scale(object.W, wnorms).^2
    W2 = object.W[:, 1:nlv].^2
    sst = zeros(nlv)
    for a = 1:nlv
        t = sqrtw .* object.T[:, a]
        tt = dot(t, t)
        beta = object.C[:, a]'
        sst[a] = tr(beta' * beta * tt)
    end 
    A = rowsum(sst' .* W2)
    B = sum(sst) * (1 / p)
    sqrt.(A / B)
end

function vip(object::Plsr, Y; nlv = nothing)
    a = nco(object.T)
    p = nro(object.W)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    W2 = object.W[:, 1:nlv].^2
    rdd = rd(Y, object.T[:, 1:nlv], object.weights)
    A = rowsum(rdd .* W2)
    B = sum(rdd) * (1 / p)
    sqrt.(A / B)
end





