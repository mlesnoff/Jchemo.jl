struct LwplsrAgg
    X::Array{Float64}
    Y::Array{Float64}
    nlvdis::Int
    metric::String
    h::Real
    k::Int
    nlv::String
    wagg::String
    tol::Real
    verbose::Bool
end

"""
    lwplsr_agg(X, Y; nlvdis, metric, h, k, nlv, verbose = false)
Aggregation of KNN-LWPLSR models with different numbers of LVs.
* `X` : X-data.
* `Y` : Y-data.
* `nlvdis` : Number of latent variables (LVs) to consider in the global PLS used for the dimension reduction before 
    calculating the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors. Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `nlv` : A character string such as "5:20" defining the range of the numbers of LVs 
    to consider ("5:20": the predictions of models with nb LVS = 5, 6, ..., 20 are averaged). 
    Syntax such as "10" is also allowed ("10": correponds to the single model with 10 LVs).
* `wagg` : Type of averaging. See function `plsr_agg`.
* `verbose` : If true, fitting information are printed.

Ensemblist method where the predictions are calculated by averaging the predictions 
of a set of KNN-LWPLSR models (`lwplsr`) built with different numbers of 
latent variables (LVs).

For instance, if argument `nlv` is set to `nlv = "5:10"`, the prediction for a new observation 
is the simple average of the predictions returned by the models with 5 LVS, 6 LVs, ... 10 LVs, respectively.
""" 
function lwplsr_agg(X, Y; nlvdis, metric, h, k, nlv, wagg = "unif", 
    tol = 1e-4, verbose = false)
    LwplsrAgg(X, Y, nlvdis, metric, h, k, nlv, wagg, 
        tol, verbose)
end

"""
    predict(object::LwplsrAgg, X)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::LwplsrAgg, X) 
    X = ensure_mat(X)
    m = size(X, 1)
    ### Getknn
    if(object.nlvdis == 0)
        res = getknn(object.X, X; k = object.k, metric = object.metric)
    else
        fm = plskern(object.X, object.Y; nlv = object.nlvdis)   
        res = getknn(fm.T, transform(fm, X); k = object.k, metric = object.metric)
        #fm = dkplsr(object.X, object.Y; nlv = object.nlvdis, gamma = 1e2)
        #res = getknn(fm.fm.T, transform(fm, X); k = object.k, metric = object.metric)
    end

    #listw = map(d -> wdist(d; h = object.h), res.d)
    listw = copy(res.d)
    for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        w[w .< object.tol] .= object.tol
        listw[i] = w
    end
    ### End
    pred = locw(object.X, object.Y, X; 
        listnn = res.ind, listw = listw, fun = plsr_agg, nlv = object.nlv, 
        wagg = object.wagg, verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end

function predict_steps(object::LwplsrAgg, X; steps = nothing) 
    X = ensure_mat(X)
    m = size(X, 1)
    ### Getknn
    if(object.nlvdis == 0)
        res = getknn(object.X, X; k = object.k, metric = object.metric)
    else
        fm = plskern(object.X, object.Y; nlv = object.nlvdis)
        res = getknn(fm.T, transform(fm, X); k = object.k, metric = object.metric)
    end
    listw = copy(res.d)
    for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        if isnothing(steps)
            w[w .< object.tol] .= object.tol
        else 
            w[1:steps] .= 1 ; w[(steps + 1):end] .= object.tol
        end
        listw[i] = w
    end
    ### End
    pred = locw(object.X, object.Y, X; 
        listnn = res.ind, listw = listw, fun = plsr_agg, nlv = object.nlv, 
        wagg = object.wagg, verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end






