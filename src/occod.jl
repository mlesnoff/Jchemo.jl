"""
    occod(object::Union{Pca, Plsr}, X; nlv = nothing, 
        typc = :mad, cri = 3, alpha = .025, kwargs...)
One-class classification using PCA/PLS orthognal distance (OD).

* `object` : The model (e.g. PCA) that was fitted on the training data,
    assumed to represent the training class.
* `X` : X-data (training) that were used to fit the model.
* `nlv` : Nb. components (PCs or LVs) to consider. If nothing, 
    it is the maximum nb. of components of the fitted model.
* `typc` : Type of cutoff (:mad or :q). See Thereafter.
* `cri` : When `typc = :mad`, a constant. See thereafter.
* `alpha` : When `typc = :q`, a risk-I level. See thereafter.
* `kwargs` : Optional arguments to pass in function `kde` of 
    KernelDensity.jl (see function `kde1`).

In this method, the outlierness `d` of an observation
is the orthogonal distance (OD =  "X-residuals") of this observation, ie.
the Euclidean distance between the observation and its projection on the 
score plan defined by the fitted (e.g. PCA) model (e.g. Hubert et al. 2005, 
Van Branden & Hubert 2005 p. 66, Varmuza & Filzmoser 2009 p. 79).

See `?occsd` for details on outputs, and examples.

## References
M. Hubert, P. J. Rousseeuw, K. Vanden Branden (2005). ROBPCA: a new approach 
to robust principal components analysis. Technometrics, 47, 64-79.

K. Vanden Branden, M. Hubert (2005). Robuts classification in high dimension based 
on the SIMCA method. Chem. Lab. Int. Syst, 79, 10-21.

K. Varmuza, P. Filzmoser (2009). Introduction to multivariate statistical analysis 
in chemometrics. CRC Press, Boca Raton.
""" 
function occod(object::Union{Pca, Plsr}, X; nlv = nothing, 
        typc = :mad, cri = 3, alpha = .025, kwargs...)
    X = ensure_mat(X)
    a = nco(object.T)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    E = xresid(object, X; nlv = nlv)
    d2 = vec(sum(E .* E, dims = 2))
    d = sqrt.(d2)
    typc == :mad ? cutoff = median(d) + cri * mad(d) : nothing
    typc == :q ? cutoff = quantile(d, 1 - alpha) : nothing
    e_cdf = StatsBase.ecdf(d)
    p_val = pval(e_cdf, d)
    d = DataFrame(d = d, dstand = d / cutoff, pval = p_val)
    Occod(d, object, e_cdf, cutoff, nlv)
end

"""
    predict(object::Occod, X)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Occod, X)
    E = xresid(object.fm, X; nlv = object.nlv)
    m = nro(E)
    d2 = vec(sum(E .* E, dims = 2))
    d = sqrt.(d2)
    p_val = pval(object.e_cdf, d)
    d = DataFrame(d = d, dstand = d / object.cutoff, 
        pval = p_val)
    pred = reshape(Int64.(d.dstand .> 1), m, 1)
    (pred = pred, d)
end


