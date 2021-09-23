"""
    scordis(object::Union{Pca, Plsr}, Xtrain, X = nothing; nlv = nothing)
Compute the score distances (SDs) from a PCA or PLS model

* `object` : The fitted model.
* `Xtrain` : X-data that were used to compute the model (training).
* `X` : X-data for which new distances are computed.
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, it is the maximum nb. of components.
* `rob` : If true, the moment estimation of the distance cutoff is robustified. 
    This may be relevant after robust PCA or PLS on small data sets containing extreme values.
* `alpha` : Risk-I level used for computing the cutoff detecting extreme values.

SDs are the Mahalanobis distances of the projections of row observations on the 
score plan to the center of the score space.

They are computed for the training `Xtrain` and the eventual `X`.

A cutoff is computed from the training, using a moment estimation of the parameters of 
a Chi-squared distrbution for SD^2 (see e.g. Pomerantzev 2008). 
In the output, column dstand is a standardized distance, defined as SD / cutoff. 
A value dstand > 1 may be considered as extreme.

The Winisi "GH" is also provided (usually considered as extreme if GH > 3).

## References

M. Hubert, P. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust 
principal components analysis. Technometrics, 47, 64-79.

Pomerantsev, A.L., 2008. Acceptance areas for multivariate classification derived by 
projection methods. Journal of Chemometrics 22, 601-609. https://doi.org/10.1002/cem.1147
""" 
function scordis(object::Union{Pca, Plsr}, X = nothing; nlv = nothing, 
    rob = false, alpha = .01)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    T = @view(object.T[:, 1:nlv])
    S = Statistics.cov(T, corrected = false)
    Sinv = inv(S) 
    d2 = mahsq(T, zeros(nlv)', Sinv)
    d = sqrt.(d2)
    if(!rob) 
        mu = mean(d2)   
        s2 = var(d2)
    else
        mu = median(d2)
        s2 = mad(d2)^2
    end
    nu = 2 * mu^2 / s2
    cutoff = sqrt(mu / nu * quantile.(Chisq(nu), 1 - alpha))
    dstand = d / cutoff 
    res_train = DataFrame((d = d, dstand = dstand, gh = d2 / nlv))
    res = nothing
    if !isnothing(X)
        T = transform(object, X; nlv = nlv)
        d2 = mahsq(T, zeros(nlv)', Sinv)
        d = sqrt.(d2)
        dstand = d / cutoff 
        res = DataFrame((d = d, dstand = dstand, gh = d2 / nlv))
    end
    (res_train = res_train, res = res, cutoff = cutoff)
end


"""
    odis(object::Union{Pca, Plsr}, Xtrain, X = nothing; nlv = nothing)
Compute the orthogonal distances (ODs) from a PCA or PLS model

* `object` : The fitted model.
* `Xtrain` : X-data that were used to compute the model (training).
* `X` : X-data for which new distances are computed.
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, it is the maximum 
    nb. of components.
* `rob` : If true, the moment estimation of the distance cutoff is robustified. 
    This may be relevant after robust PCA or PLS on small data sets containing extreme values.
* `alpha` : Risk-I level used for computing the cutoff detecting extreme values.

ODs ("X-residuals") are the Euclidean distances of row observations to their projections 
to the score plan (see e.g. Hubert et al. 2005, Van Branden & Hubert 2005, p. 66; 
Varmuza & Filzmoser, 2009, p. 79).

They are computed for the training `Xtrain` and the eventual `X`.

A cutoff is computed from the training, using a moment estimation of the parameters of a 
Chi-squared distrbution for OD^2 (see Nomikos & MacGregor 1995, and Pomerantzev 2008). 
In the output, column dstand is a standardized distance, defined as OD / cutoff. 
A value dstand > 1 may be considered as extreme.

The cutoff for detecting extreme OD values is computed using a moment estimation of 
a Chi-squared distrbution for the squared distance.

## References

M. Hubert, P. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust principal 
components analysis. Technometrics, 47, 64-79.

Nomikos, P., MacGregor, J.F., 1995. Multivariate SPC Charts for Monitoring Batch Processes. 
null 37, 41-59. https://doi.org/10.1080/00401706.1995.10485888

Pomerantsev, A.L., 2008. Acceptance areas for multivariate classification derived by projection methods. 
Journal of Chemometrics 22, 601â-609. https://doi.org/10.1002/cem.1147

K. Vanden Branden, M. Hubert (2005). Robuts classification in high dimension based on the SIMCA method. 
Chem. Lab. Int. Syst, 79, 10-21.

K. Varmuza, P. Filzmoser (2009). Introduction to multivariate statistical analysis in chemometrics. 
CRC Press, Boca Raton.
""" 
function odis(object::Union{Pca, Plsr}, Xtrain, X = nothing; nlv = nothing, 
        rob = false, alpha = .01)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    E = xresid(object, Xtrain; nlv = nlv)
    d = sqrt.(sum(E .* E, dims = 2))
    d2 = d.^2
    if(!rob) 
        mu = mean(d2)   
        s2 = var(d2)
    else
        mu = median(d2)
        s2 = mad(d2)^2
    end
    nu = 2 * mu^2 / s2
    cutoff = sqrt(mu / nu * quantile.(Chisq(nu), 1 - alpha))
    dstand = d / cutoff 
    res_train = DataFrame((d = d, dstand = dstand))
    res = nothing
    if !isnothing(X)
        E = xresid(object, X; nlv = nlv)
        d = sqrt.(sum(E .* E, dims = 2))
        dstand = d / cutoff 
        res = DataFrame((d = d, dstand = dstand))
    end
    (res_train = res_train, res = res, cutoff = cutoff)
end

