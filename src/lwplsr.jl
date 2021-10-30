struct Lwplsr
    X::Array{Float64}
    Y::Array{Float64}
    fm
    metric::String
    h::Real
    k::Int
    nlv::Int
    tol::Real
    verbose::Bool
end

"""
    lwplsr(X, Y; nlvdis, metric, h, k, nlv, verbose = false)
k-Nearest-Neighbours locally weighted partial least squares regression (KNN-LWPLSR).
* `X` : X-data.
* `Y` : Y-data.
* `nlvdis` : Number of latent variables (LVs) to consider in the global PLS used for the dimension reduction before 
    calculating the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : Type of dissimilarity used to select the neighbors. Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `nlv` : Nb. latent variables (LVs).
* `verbose` : If true, fitting information are printed.

Function `lwplsr` fits KNN-LWPLSR models (Lesnoff et al., 2020). The function uses functions `getknn`,
`locw` and a PLSR function; see the code for details. Many other variants of KNN-LWPLSR pipelines can be build.

The general principles of the method are as follows.

LWPLSR is a particular case of weighted PLSR (WPLSR; e.g. Schaal et al. 2002). 
In WPLSR, a priori weights, different from the usual 1/n (standard PLSR), are given to the n training observations. 
These weights are used for calculating (i) the PLS scores and loadings and (ii) the regression model 
of the response(s) over the scores (by weighted least squares). 
The specificity of LWPLSR is that the weights are computed from dissimilarities (e.g. distances) 
between the new observation to predict and the training observations ("L" comes from "localized"). 
A characteristic of LWPLSR is that the weights, and therefore the fitted WPLSR model, 
change for each new observation to predict.

Basic versions of LWPLSR use, for each observation to predict, all the n training observation 
(e.g. Sicard & Sabatier 2006, Kim et al 2011). This can be very time consuming, 
in particular for large n. A faster and often more efficient strategy is to preliminary select, 
in the training set, a number of k nearest neighbors to the observation to predict 
("weighting 1") and then to apply LWPLSR only to this pre-selected neighborhood ("weighting 2"). 
This strategy corresponds to a kNN-LWPLSR and is the one implemented in function `lwplsr`.

In `lwplsr`, the dissimilarities used for computing the weights can be 
calculated from the original X-data or after a dimension reduction (depending on argument `nlvdis`). 
In the last case, global PLS scores (LVs) are computed from (`X`, `Y`) and the dissimilarities are 
calculated over these scores. In general, for high dimensional X-data, using the Mahalanobis distance requires 
preliminary dimensionality reduction of the data.

## References

Kim, S., Kano, M., Nakagawa, H., Hasebe, S., 2011. Estimation of active pharmaceutical ingredients 
content using locally weighted partial least squares and statistical wavelength selection. Int. J. Pharm., 421, 269-274.

Lesnoff, M., Metz, M., Roger, J.-M., 2020. Comparison of locally weighted PLS 
strategies for regression and discrimination on agronomic NIR data. Journal of Chemometrics, e3209. https://doi.org/10.1002/cem.3209

Schaal, S., Atkeson, C., Vijayamakumar, S. 2002. Scalable techniques from nonparametric 
statistics for the real time robot learning. Applied Intell., 17, 49-60.

Sicard, E. Sabatier, R., 2006. Theoretical framework for local PLS1 regression 
and application to a rainfall data set. Comput. Stat. Data Anal., 51, 1393-1410.

""" 
function lwplsr(X, Y; nlvdis, metric, h, k, nlv, tol = 1e-4, verbose = false)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    if nlvdis == 0
        fm = nothing
    else
        fm = plskern(X, Y; nlv = nlvdis)
    end
    return Lwplsr(X, Y, fm, metric, h, k, nlv, tol, verbose)
end

"""
    predict(object::Lwplsr, X)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Lwplsr, X; nlv = nothing)
    X = ensure_mat(X)
    m = size(X, 1)
    a = object.nlv
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    # Getknn
    if isnothing(object.fm)
        res = getknn(object.X, X; k = object.k, metric = object.metric)
    else
        res = getknn(object.fm.T, transform(object.fm, X); k = object.k, metric = object.metric) 
    end
    listw = copy(res.d)
    for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        w[w .< object.tol] .= object.tol
        listw[i] = w
    end
    # End
    pred = locwlv(object.X, object.Y, X; 
        listnn = res.ind, listw = listw, fun = plskern, nlv = nlv, 
        verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end



