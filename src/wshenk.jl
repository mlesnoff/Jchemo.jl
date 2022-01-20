"""
    wshenk(object::Union{Pca, Plsr}, X; nlv = nothing)
Compute the Shenk et al. (1997) PLSR weights
* `object` : The fitted model.
* `X` : X-data on which the weights are computed.
* `nlv` : Nb. latent variables (LVs) to consider. If nothing, 
    it is the maximum nb. of components.

For each observation (row) of `X`, the weights are returned 
for the models with 1, ..., nlv LVs. 

## References

Shenk, J., Westerhaus, M., Berzaghi, P., 1997. Investigation of a LOCAL calibration 
procedure for near infrared instruments. 
Journal of Near Infrared Spectroscopy 5, 223. https://doi.org/10.1255/jnirs.115

""" 
function wshenk(object::Union{Plsr}, X; nlv = nothing)
    X = ensure_mat(X)
    m = size(X, 1)
    p = size(X, 2)
    q = size(object.C, 1)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    E = similar(X)
    rms_r = similar(X, m, a)
    zB = similar(X, p + 1, q)
    rms_b = similar(X, a)
    W = similar(X, m, a)
    for j = 1:a
        E .= xresid(object, X, nlv = j)
        rms_r[:, j] .= vec(sqrt.(sum(E.^2, dims = 2)))
        z = coef(object; nlv = j)
        zB[1, :] .= vec(z.int)
        zB[2:end, :] .= z.B
        rms_b[j] = sqrt(mean(mean(zB.^2, dims = 2))) 
        W[:, j] .= 1 ./ (vcol(rms_r, j) * rms_b[j])
    end 
    for i = 1:m
        W[i, :] ./= sum(vrow(W, i)) 
    end
    (W = W, rms_r = rms_r, rms_b = rms_b)
end

