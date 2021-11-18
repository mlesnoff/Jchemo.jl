struct Plsrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

struct KplsrDa
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

"""
    plsrda(X, y, weights = ones(size(X, 1)); nlv)
Discrimination (DA) based on partial least squares regression (PLSR).
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights of the observations.
* `nlv` : Nb. latent variables (LVs) to compute.

This is the common "PLSDA". 
The training variable y (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy variable (0/1). 
Then, a PLS2 is implemented on the X-data and Ydummy, 
returning `nlv` latent variables (LVs).
Finally, a multiple linear regression (MLR) is run between the LVs and each 
column of Ydummy, returning predictions of the dummy variables 
(= object `posterior` returned by function `predict`). 
These predictions can be considered as unbounded (i.e. eventuall outside of [0, 1]) 
estimates of the class membership probabilities.
For a given observation, the final prediction is the class corresponding 
to the dummy variable for which the prediction is the highest.
""" 
function plsrda(X, y, weights = ones(size(X, 1)); nlv)
    z = dummy(y)
    fm = plskern(X, z.Y, weights; nlv = nlv)
    Plsrda(fm, z.lev, z.ni)
end

"""
    kplsrda(X, y, weights = ones(size(X, 1)); nlv, kern = "krbf", kwargs...)
Discrimination (DA) based on kernel partial least squares regression (KPLSR).
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations.
* `nlv` : Nb. latent variables (LVs) to compute.

This is the common "PLSDA". 
The training variable y (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy variable (0/1). 
Then, a kernel PLS2 is implemented on the X-data and Ydummy, 
returning `nlv` latent variables (LVs).
Finally, a multiple linear regression (MLR) is run between the LVs and each 
column of Ydummy, returning predictions of the dummy variables 
(= object `posterior` returned by function `predict`). 
These predictions can be considered as unbounded (i.e. eventuall outside of [0, 1]) 
estimates of the class membership probabilities.
For a given observation, the final prediction is the class corresponding 
to the dummy variable for which the prediction is the highest.
""" 
function kplsrda(X, y, weights = ones(size(X, 1)); nlv, kern = "krbf", kwargs...)
    z = dummy(y)
    fm = kplsr(X, z.Y, weights; nlv = nlv, kern = kern, kwargs...)
    KplsrDa(fm, z.lev, z.ni)
end

"""
    predict(object::Union{Plsrda, KplsrDa}, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
    If nothing, it is the maximum nb. LVs.
""" 
function predict(object::Union{Plsrda, KplsrDa}, X; nlv = nothing)
    X = ensure_mat(X)
    m = size(X, 1)
    a = size(object.fm.T, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(le_nlv)
    posterior = list(le_nlv)
    @inbounds for i = 1:le_nlv
        zp = predict(object.fm, X; nlv = nlv[i]).pred
        #if softmax
        #    @inbounds for j = 1:m
        #        zp[j, :] .= mweights(exp.(zp[j, :]))
        #   end
        #end
        z =  mapslices(argmax, zp; dims = 2)  # if equal, argmax takes the first
        pred[i] = reshape(replacebylev(z, object.lev), m, 1)
        posterior[i] = zp
    end 
    if le_nlv == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end

