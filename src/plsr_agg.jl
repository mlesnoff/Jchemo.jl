struct PlsrAgg
    fm::Plsr
    nlv
end

""" 
    plsr_agg(X, Y, weights = ones(size(X, 1)) ; nlv)
    plsr_agg!(X, Y, weights = ones(size(X, 1)) ; nlv)
Ensemblist approach where the predictions are calculated by averaging 
the predictions of PLSR models (plskern) built with different numbers of latent variables (LVs).
- X : matrix (n, p), or vector (n,).
- Y : matrix (n, q), or vector (n,).
- weights : vector (n,).
- nlv : A character string such as "5:20" defining the range of the numbers of LVs to consider. 
Syntax such as "10" is also allowed.

For instance, if argument nlv is set to nlv = "5:10", 
the prediction for a new observation will be the average (without weighting) 
of the predictions returned by the models with 5 LVS, 6 LVs, ... 10 LVs.
If nlv = "10", this will be the prediction of the single model with 10 LVs.

X and Y are internally centered. 
The inplace version modifies externally X and Y. 
""" 
function plsr_agg(X, Y, weights = ones(size(X, 1)) ; nlv)
    res = plsr_agg!(copy(X), copy(Y), weights; nlv = nlv)
    res
end

function plsr_agg!(X, Y, weights = ones(size(X, 1)) ; nlv)
    n = size(X, 1)
    p = size(X, 2)
    nlv = eval(Meta.parse(nlv))
    nlv = (max(minimum(nlv), 0):min(maximum(nlv), n, p))
    fm = plskern!(X, Y, weights; nlv = maximum(nlv))
    PlsrAgg(fm, nlv)
end

"""
    predict(object::PlsrAgg, X)
Compute Y-predictions from the fitted model(s).
- object : The fitted model(s).
- X : Matrix (m, p) for which predictions are computed.
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


