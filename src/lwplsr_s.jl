struct LwplsrS1
    X::Array{Float64}
    Y::Array{Float64}
    fm0
    fm
    nlv0::Int
    nlvdis::Int
    metric::String
    h::Real
    k::Int
    nlv::Int
    tol::Real
    verbose::Bool
end

"""
    lwplsr_s(X, Y; nlvdis, metric, h, k, nlv, tol = 1e-4, verbose = false)
kNN-LWPLSR after preliminary dimension reduction.
* `X` : X-data.
* `Y` : Y-data.
* `nlv0` : Nb. latent variables (LVs) for preliminary data reduction. 
* `nlvdis` : Nb. LVs to consider in the global PLS 
    used for the dimension reduction before computing the dissimilarities. 
    If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors. 
    Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, 
    sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `nlv` : Nb. latent variables (LVs).
* `tol` : For stabilization when very close neighbors.
* `verbose` : If true, fitting information are printed.

This is a fast version of kNN-LWPLSR (Lesnoff et al. 2020) using the same principle as 
in Shen et al 2019. The kNN-LWPLSR is done after preliminary dimension reduction
(parameter `nlv0`) of the X-data.

## References
Lesnoff, M., Metz, M., Roger, J.-M., 2020. Comparison of locally weighted PLS 
strategies for regression and discrimination on agronomic NIR data. 
Journal of Chemometrics, e3209. https://doi.org/10.1002/cem.3209

Shen, G., Lesnoff, M., Baeten, V., Dardenne, P., Davrieux, F., Ceballos, H., Belalcazar, J., 
Dufour, D., Yang, Z., Han, L., Pierna, J.A.F., 2019. Local partial least squares based on global PLS scores. 
Journal of Chemometrics 0, e3117. https://doi.org/10.1002/cem.3117
""" 
function lwplsr_s(X, Y; nlv0,
        nlvdis, metric, h, k, nlv, tol = 1e-4, verbose = false)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    fm0 = plskern(X, Y; nlv = nlv0)
    if nlvdis == 0
        fm = nothing
    else
        fm = plskern(fm0.T, Y; nlv = nlvdis)
    end
    LwplsrS1(X, Y, fm0, fm, nlv0, nlvdis, metric, h, k, nlv, 
        tol, verbose)
end

"""
    predict(object::Lwplsr, X)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::LwplsrS1, X; nlv = nothing)
    X = ensure_mat(X)
    m = size(X, 1)
    a = object.nlv
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    T = transform(object.fm0, X)
    # Getknn
    if isnothing(object.fm)
        res = getknn(object.fm0.T, T; k = object.k, metric = object.metric)
    else
        res = getknn(object.fm.T, transform(object.fm, T); k = object.k, metric = object.metric) 
    end
    listw = copy(res.d)
    for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        w[w .< object.tol] .= object.tol
        listw[i] = w
    end
    # End
    pred = locwlv(object.fm0.T, object.Y, T; 
        listnn = res.ind, listw = listw, fun = plskern, nlv = nlv, 
        verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end

