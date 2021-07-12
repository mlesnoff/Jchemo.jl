struct PlsrAgg
    fm::Plsr
    nlv
end

""" 
    plsr_agg(X, Y, weights = ones(size(X, 1)); nlv)
Aggregation of PLSR models with different numbers of LVs.
* `X` : X-data.
* `Y` : Y-data.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 are averaged). 
    Syntax such as "10" is also allowed ("10": correponds to the single model with 10 LVs).
* `verbose` : If true, fitting information are printed.

Ensemblist method where the predictions are calculated by averaging the predictions 
of a set of PLSR models (`plskern`) built with different numbers of latent variables (LVs).

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for a new observation 
is the simple average of the predictions returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.
""" 
function plsr_agg(X, Y, weights = ones(size(X, 1)); nlv)
    res = plsr_agg!(copy(X), copy(Y), weights; nlv = nlv)
    res
end

function plsr_agg!(X, Y, weights = ones(size(X, 1)); nlv)
    n = size(X, 1)
    p = size(X, 2)
    nlv = eval(Meta.parse(nlv))
    nlv = (max(minimum(nlv), 0):min(maximum(nlv), n, p))
    fm = plskern!(X, Y, weights; nlv = maximum(nlv))
    PlsrAgg(fm, nlv)
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
        acc = copy(zpred[1])
        for i = 2:le_nlv
            acc .+= zpred[i]
        end
        pred = acc ./ le_nlv
    end
    (pred = pred, predlv = zpred)
end


