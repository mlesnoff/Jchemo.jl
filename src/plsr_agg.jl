struct PlsrAgg
    fm::Plsr
    nlv
    wagg
    w
end

""" 
    plsr_agg(X, Y, weights = ones(size(X, 1)); nlv, wagg = "unif")
Aggregation of PLSR models with different numbers of LVs.
* `X` : X-data.
* `Y` : Y-data. Must be univariate if `wagg` != "unif".
* weights : Weights of the observations.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 are averaged). 
    Syntax such as "10" is also allowed ("10": correponds to the single model with 10 LVs).
* `wagg` : Type of averaging. 

Ensemblist method where the predictions are calculated by averaging the predictions 
of a set of PLSR models (`plskern`) built with different numbers of latent variables (LVs).

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for a new observation 
is the average (eventually weighted) of the predictions returned by the models 
with 5 LVS, 6 LVs, ... 10 LVs, respectively.

Dependending argument `wagg`, the average is the simple mean (`wagg` = "unif") or 
a weighted mean using AIC weights computed from the models (see function `aicplsr`):
* "aic" : Usual AIC weights (exp(-delta/2)).
* "sqrt" : Sqrt(delta) is used in place of delta.
* "fair" : A decreasing "fair" function is applied to delta.
* "inv" : Weights are computed by 1 / AIC.
""" 
function plsr_agg(X, Y, weights = ones(size(X, 1)); nlv, wagg = "unif")
    plsr_agg!(copy(X), copy(Y), weights; nlv = nlv, wagg = wagg)
end

function plsr_agg!(X, Y, weights = ones(size(X, 1)); nlv, wagg = "unif")
    n = size(X, 1)
    p = size(X, 2)
    nlv = eval(Meta.parse(nlv))
    nlv = (max(minimum(nlv), 0):min(maximum(nlv), n, p))
    if isequal(wagg, "unif")
        w = ones(maximum(nlv) + 1)
    elseif isequal(wagg, "aic")
        w = aicplsr(X, Y; nlv = maximum(nlv)).w.aic
    elseif isequal(wagg, "sqrt")
        d = aicplsr(X, Y; nlv = maximum(nlv)).delta.aic
        w = exp.(-sqrt.(d) / 2)
    elseif isequal(wagg, "fair")
        d = aicplsr(X, Y; nlv = maximum(nlv)).delta.aic
        d = d / maximum(d[isnan.(d) .== 0])
        w = 1 ./ (1 .+ d).^2
    elseif isequal(wagg, "inv")
        w = 1 ./ aicplsr(X, Y; nlv = maximum(nlv)).crit.aic
    end
    w[isnan.(w)] .= 0
    #w = vec(mavg(w', 3))
    w = w[collect(nlv) .+ 1]
    w = w / sum(w)
    fm = plskern!(X, Y, weights; nlv = maximum(nlv))
    PlsrAgg(fm, nlv, wagg, w)
end

"""
    predict(object::PlsrAgg, X)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::PlsrAgg, X)
    nlv = object.nlv
    le_nlv = length(nlv)
    zpred = predict(object.fm, X; nlv = nlv).pred
    if(le_nlv == 1)
        pred = zpred
    else
        acc = object.w[1] * copy(zpred[1])
        @inbounds for i = 2:le_nlv
            acc .+= object.w[i] * zpred[i]
        end
        pred = acc
    end
    (pred = pred, predlv = zpred)
end


