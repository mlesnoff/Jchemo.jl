struct Dkplsr
    X::Array{Float64}
    fm
    K::Array{Float64}
    kern
    dots
end

"""
    dkplsr(X, Y, weights = ones(size(X, 1)); nlv , kern = "krbf", kwargs...)
Direct Kernel Partial Least Squares regression (DKPLSR) (Bennett & Embrechts 2003).

* `X` : matrix (n, p), or vector (n,).
* `Y` : matrix (n, q), or vector (n,).
* `weights` : vector (n,).
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
* 'kern' : Type of kernel function used to compute the Gram matrices.
    Possible values are "krbf" of "kpol" (see respective functions `krbf` and `kpol`).
* `kwargs` : Named arguments to pass in the kernel function.

The method builds kernel Gram matrices and then runs a usual PLSR algorithm on them. This is faster 
(but not equivalent) to the "true" NIPALS KPLSR algorithm such as described in Rosipal & Trejo (2001).

The in-place version modifies `Y`.

## References 

Bennett, K.P., Embrechts, M.J., 2003. An optimization perspective on kernel partial least squares regression, 
in: Advances in Learning Theory: Methods, Models and Applications, 
NATO Science Series III: Computer & Systems Sciences. IOS Press Amsterdam, pp. 227-250.

Rosipal, R., Trejo, L.J., 2001. Kernel Partial Least Squares Regression in Reproducing Kernel Hilbert Space. 
Journal of Machine Learning Research 2, 97-123.

""" 
function dkplsr(X, Y, weights = ones(size(X, 1)); nlv, kern = "krbf", kwargs...)
    dkplsr!(copy(X), copy(Y), weights; nlv = nlv, kern = kern, kwargs...)
end

function dkplsr!(X, Y, weights = ones(size(X, 1)); nlv, kern = "krbf", kwargs...)
    fkern = eval(Meta.parse(kern))    
    K = fkern(X, X; kwargs...)     # In the future: fkern!(K, X, X; kwargs...)
    fm = plskern!(K, Y; nlv = nlv)
    Dkplsr(X, fm, K, kern, kwargs)
end

""" 
    transform(object::Dkplsr, X; nlv = nothing)
Compute LVs ("scores" T) from a fitted model and a matrix X.
* `object` : The maximal fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider. If nothing, it is the maximum nb. LVs.
""" 
function transform(object::Dkplsr, X; nlv = nothing)
    fkern = eval(Meta.parse(object.kern))
    K = fkern(X, object.X; object.dots...)
    transform(object.fm, K; nlv = nlv)
end

"""
    coef(object::Dkplsr; nlv = nothing)
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
If nothing, it is the maximum nb. LVs.
""" 
function coef(object::Dkplsr; nlv = nothing)
    coef(object.fm; nlv = nlv)
end

"""
    predict(object::Dkplsr, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The maximal fitted model.
* `X` : Matrix (m, p) for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
If nothing, it is the maximum nb. LVs.
""" 
function predict(object::Dkplsr, X; nlv = nothing)
    fkern = eval(Meta.parse(object.kern))
    K = fkern(X, object.X; object.dots...)
    pred = predict(object.fm, K; nlv = nlv).pred
    (pred = pred,)
end
