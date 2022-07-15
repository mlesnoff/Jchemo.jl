struct Scordis
    dis
    fm
    Sinv::Matrix{Float64}
    cutoff::Real   
    nlv::Int64
end

struct Odis
    dis
    fm
    cutoff::Real   
    nlv::Int64
end

"""
    scordis(object::Union{Pca, Plsr}; nlv = nothing, rob = true, alpha = .01)
Compute the score distances (SDs) from a PCA or PLS model

* `object` : The fitted model.
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, 
    it is the maximum nb. of components of the fitted model.
* `rob` : If true, the moment estimation of the distance cutoff is robustified. 
    This may be relevant after robust PCA or PLS on small data sets 
        containing extreme values.
* `alpha` : Risk-I level used for computing the cutoff detecting extreme values.

SDs are the Mahalanobis distances of the projected row observations on the 
score plan to the center of the score plan.

A cutoff is computed from the training, using a moment estimation of 
the Chi-squared distrbution assumed for SD^2 (see e.g. Pomerantzev 2008). 
In the output, column `dstand` is a standardized distance defined as SD / cutoff. 
A value dstand > 1 may be considered as extreme.

The Winisi "GH" is also provided (usually, GH > 3 is considered as extreme).

## References
M. Hubert, P. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach to robust 
principal components analysis. Technometrics, 47, 64-79.

Pomerantsev, A.L., 2008. Acceptance areas for multivariate classification derived by 
projection methods. Journal of Chemometrics 22, 601-609. https://doi.org/10.1002/cem.1147

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlv = 15 
fm = plskern(Xtrain, ytrain; nlv = nlv) ;

res = scordis(fm; nlv = nlv) 
pnames(res)
sdtrain = res.dis
sdtest = predict(res, Xtest).dis

res = odis(fm, Xtrain; nlv = nlv)
pnames(res)
odtrain = res.dis
odtest = predict(res, Xtest).dis

f, ax = scatter(sdtrain.dstand, odtrain.dstand,
    axis = (xlabel = "SD", ylabel = "OD"), label = "Train")
scatter!(ax, sdtest.dstand, odtest.dstand, color = (:red, .5), label = "Test")
hlines!(ax, 1; color = :grey, linestyle = "-")
vlines!(ax, 1; color = :grey, linestyle = "-")
axislegend(position = :rt)
f
```
""" 
function scordis(object::Union{Pca, Plsr}; 
        nlv = nothing, rob = true, alpha = .01)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    T = @view(object.T[:, 1:nlv])
    S = Statistics.cov(T, corrected = false)
    LinearAlgebra.inv!(cholesky!(S))   # ==> S := Sinv
    d2 = mahsq(T, zeros(nlv)', S)
    d = sqrt.(d2)
    if !rob 
        mu = mean(d2)   
        s2 = var(d2)
    else
        mu = median(d2)
        s2 = mad(d2)^2
    end
    nu = 2 * mu^2 / s2
    cutoff = sqrt(mu / nu * quantile.(Distributions.Chisq(nu), 1 - alpha))
    dstand = d / cutoff 
    dis = DataFrame((d = d, dstand = dstand, gh = d2 / nlv))
    Scordis(dis, object, S, cutoff, nlv)
end

function predict(object::Scordis, X)
    nlv = object.nlv
    T = transform(object.fm, X; nlv = nlv)
    d2 = mahsq(T, zeros(nlv)', object.Sinv)
    d = sqrt.(d2)
    dstand = d / object.cutoff 
    dis = DataFrame((d = d, dstand = dstand, gh = d2 / nlv))
    (dis = dis,)
end

"""
    odis(object::Union{Pca, Plsr}, X; nlv = nothing, rob = true, alpha = .01)
Compute the orthogonal distances (ODs) from a PCA or PLS model

* `object` : The fitted model.
* `X` : X-data that were used to compute the model (training).
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, it is the maximum 
    nb. of components.
* `rob` : If true, the moment estimation of the distance cutoff is robustified. 
    This may be relevant after robust PCA or PLS on small data sets 
        containing extreme values.
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

See `?scordis` for examples.

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
function odis(object::Union{Pca, Plsr}, X; 
        nlv = nothing, rob = true, alpha = .01)
    X = ensure_mat(X)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    E = xresid(object, X; nlv = nlv)
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
    cutoff = sqrt(mu / nu * quantile.(Distributions.Chisq(nu), 1 - alpha))
    dstand = d / cutoff 
    dis = DataFrame((d = d, dstand = dstand))
    Odis(dis, object, cutoff, nlv)
end

function predict(object::Odis, X)
    nlv = object.nlv
    E = xresid(object.fm, X; nlv = nlv)
    d = sqrt.(sum(E .* E, dims = 2))
    dstand = d / object.cutoff 
    dis = DataFrame((d = d, dstand = dstand))
    (dis = dis,)
end


