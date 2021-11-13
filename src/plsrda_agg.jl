struct PlsdaAgg
    fm
    nlv
    w_mod
end

""" 
    plsrda_agg(X, y, weights = ones(size(X, 1)); nlv)
Aggregation of PLSR-DA models with different numbers of LVs.
* `X` : X-data.
* `y` : y-data (class membership).
* weights : Weights of the observations.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 
    are averaged). Syntax such as "10" is also allowed ("10": correponds to
    the single model with 10 LVs).

Ensemblist method where the predictions are calculated by "averaging" 
the predictions of a set of models built with different numbers of 
latent variables (LVs).

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for 
a new observation is the most occurent class within the predictions 
returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.
""" 
function plsrda_agg(X, y, weights = ones(size(X, 1)); nlv)
    n = size(X, 1)
    p = size(X, 2)
    nlv = eval(Meta.parse(nlv))
    nlvmax = maximum(nlv)
    nlv = (max(minimum(nlv), 0):min(nlvmax, n, p))
    w = ones(nlvmax + 1)
    w_mod = mweights(w[collect(nlv) .+ 1])   # uniform weights
    fm = plsrda(X, y, weights; nlv = nlvmax)
    PlsdaAgg(fm, nlv, w_mod)
end

""" 
    plslda_agg(X, y, weights = ones(size(X, 1)); nlv)
Aggregation of PLSR models with different numbers of LVs.
* `X` : X-data.
* `y` : y-data (class membership).
* weights : Weights of the observations.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 
    are averaged). Syntax such as "10" is also allowed ("10": correponds to
    the single model with 10 LVs).

Ensemblist method where the predictions are calculated by "averaging" 
the predictions of a set of models built with different numbers of 
latent variables (LVs).

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for 
a new observation is the most occurent class within the predictions 
returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.
""" 
function plslda_agg(X, y, weights = ones(size(X, 1)); nlv)
    n = size(X, 1)
    p = size(X, 2)
    nlv = eval(Meta.parse(nlv))
    nlvmax = maximum(nlv)
    nlv = (max(minimum(nlv), 0):min(nlvmax, n, p))
    w = ones(nlvmax + 1)
    w_mod = mweights(w[collect(nlv) .+ 1])   # uniform weights
    fm = plslda(X, y, weights; nlv = nlvmax)
    PlsdaAgg(fm, nlv, w_mod)
end

""" 
    plsqda_agg(X, y, weights = ones(size(X, 1)); nlv)
Aggregation of PLSR models with different numbers of LVs.
* `X` : X-data.
* `y` : y-data (class membership).
* weights : Weights of the observations.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 
    are averaged). Syntax such as "10" is also allowed ("10": correponds to
    the single model with 10 LVs).

Ensemblist method where the predictions are calculated by "averaging" 
the predictions of a set of models built with different numbers of 
latent variables (LVs).

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for 
a new observation is the most occurent class within the predictions 
returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.
""" 
function plsqda_agg(X, y, weights = ones(size(X, 1)); nlv)
    n = size(X, 1)
    p = size(X, 2)
    nlv = eval(Meta.parse(nlv))
    nlvmax = maximum(nlv)
    nlv = (max(minimum(nlv), 0):min(nlvmax, n, p))
    w = ones(nlvmax + 1)
    w_mod = mweights(w[collect(nlv) .+ 1])   # uniform weights
    fm = plsqda(X, y, weights; nlv = nlvmax)
    PlsdaAgg(fm, nlv, w_mod)
end

"""
    predict(object::PlsrAgg, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::PlsdaAgg, X)
    X = ensure_mat(X)
    m = size(X, 1)
    nlv = object.nlv
    le_nlv = length(nlv)
    zpred = predict(object.fm, X; nlv = nlv).pred
    if(le_nlv == 1)
        pred = zpred
    else
        z = reduce(hcat, zpred)
        pred = similar(object.fm.lev, m, 1)
        @inbounds for i = 1:m    
            pred[i, :] .= findmax_cla(z[i, :], object.w_mod)
        end
    end
    (pred = pred, predlv = zpred)
end


