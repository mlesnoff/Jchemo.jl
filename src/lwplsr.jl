## lwplsr: Only 1 combination for arguments (including nlv)
## i.e. all arguments other than nlv must have length = 1

struct Lwplsr
    X::Array{Float64}
    Y::Array{Float64}
    nlvdis::Int
    metric::String
    h::Real
    k::Int
    nlv::Int
    verbose::Bool
end

"""
    lwplsr(X, Y; nlvdis, metric, h, k, nlv, verbose = false)
K-Nearest-Neighbours Locally Weighted Partial Least Squared Regression (KNN-LWPLSR).
* `X` : matrix (n, p), or vector (n,).
* `Y` : matrix (n, q), or vector (n,).
* `nlvdis` : The number of LVs to consider in the global PLS used for the dimension reduction before 
    calculating the dissimilarities. If `nlvdis = 0`, there is no dimension reduction.
* `metric` : The type of dissimilarity used for defining the neighbors. Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scale scalar defining the shape of the weight function. Lower is h, sharper is the function. See `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `nlv` : Nb. latent variables (LVs).
* `verbose` : If true, fitting information are printed.

Function `lwplsr` fits KNN-LWPLSR models (Lesnoff et al., 2020). The function uses functions `getknn`,
`locw` and a PLSR function; see the code for details. Many other variants of KNN-LWPLSR pipelines can be build.

The general principles of the method are as follows.

**LWPLSR** is a particular case of weighted PLSR (WPLSR) (e.g. Schaal et al. 2002). 
In WPLSR, a priori weights, different from the usual *1/n* (standard PLSR), are given to the *n* training observations. 
These weights are used for calculating (i) the PLS scores and loadings and (ii) the regression model 
of the response(s) over the scores (by weighted least squares). 
In LWPLSR, the weights are defined from dissimilarities (e.g. distances) between the new observation 
to predict and the training observations ("L" comes from "localized"). By definition of LWPLSR, the weights, 
and therefore the fitted WPLSR model, change for each new observation to predict.

Basic versions of LWPLSR (e.g. Sicard & Sabatier 2006, Kim et al 2011) use, for 
each observation to predict, all the *n* training observation. This can be very time consuming, 
in particular for large *n*. A faster and often more efficient strategy is to preliminary select, 
in the training set, a number of *k* nearest neighbors to the observation to predict 
(this is referred to as "weighting 1" in function `locw`) and then to apply LWPLSR 
only to this pre-selected neighborhood (which is referred to as weighting "2" in locw). 
This strategy corresponds to **KNN-LWPLSR** such as implemented in function `lwplsr`.

In `lwplsr`, the dissimilarities used for computing the weights can be 
calculated from the original X-data or after a dimension reduction (argument `nlvdis`). 
In the last case, global PLS scores (LVs) are computed from (X, Y) and the dissimilarities are 
calculated on these scores. For high dimensional X-data, using the Mahalanobis distance often requires 
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
function lwplsr(X, Y; nlvdis, metric, h, k, nlv, verbose = false)
    return Lwplsr(X, Y, nlvdis, metric, h, k, nlv, verbose)
end

function predict(object::Lwplsr, X; nlv = nothing) 
    a = object.nlv
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    ## Getknn
    if(object.nlvdis == 0)
        res = getknn(object.X, X; k = object.k, metric = object.metric)
    else
        fm = plskern(object.X, object.Y; nlv = object.nlvdis)
        res = getknn(fm.T, transform(fm, X); k = object.k, metric = object.metric)
    end
    listw = map(d -> wdist(d, object.h), res.d)
    ## End
    pred = locwlv(object.X, object.Y, X; 
        listnn = res.ind, listw = listw, fun = plskern, nlv = nlv, verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, listw = listw)
end



