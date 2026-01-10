"""
    umap(; kwargs...)
    umap(X; kwargs...)
UMAP: Uniform manifold approximation and projection for dimension reduction
* `X` : X-data (n, p).
Keyword arguments:
* `psamp` : Proportion of sampling in `X` for training.
* `nlv` : Nb. latent variables (LVs) to compute.
* `n_neighbors` : Nb. approximate neighbors used to construct the initial high-dimensional graph.
* `min_dist` : Minimum distance between points in low-dimensional space.
* `scal` : Boolean. If `true`, each column of `X` and `Y` is scaled by its uncorrected standard deviation.
    
The function fits a UMAP dimension reducion using package `UMAP.jl'. The used metric is the Euclidean distance. 

If `psamp < 1`, only a proportion `psamp` of the observations (rows of `X`) are used to build the model (systematic 
sampling over the first score of the PCA of `X`). Can be used to decrease computation times when n is large.

## References

https://github.com/dillondaudert/UMAP.jl

McInnes, L, Healy, J, Melville, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. 
ArXiV 1802.03426, 2018. https://arxiv.org/abs/1802.03426

https://umap-learn.readthedocs.io/en/latest/how_umap_works.html

https://pair-code.github.io/understanding-umap/ 

## Examples
```julia
```
""" 
umap(; kwargs...) = JchemoModel(umap, nothing, kwargs)

function umap(X; kwargs...)
    par = recovkw(ParUmap, kwargs).par
    X = ensure_mat(X)
    Q = eltype(X)
    n, p = size(X)
    if par.psamp < 1
        ns = round(Int, par.psamp * n)
        res = nipals(fcenter(X, colmean(X)); maxit = 50)
        s = sampsys(res.u, ns).test
        X = vrow(X, s)
    else
        s = collect(1:n)
    end
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X)
        X = fscale(X, xscales)
    end
    ## Note: UMAP.jl ==> the type of new_data must match the original data exactly ==> force to Matrix
    fitm = UMAP.fit(Matrix(X'), par.nlv; metric = par.metric, n_neighbors = par.n_neighbors, min_dist = par.min_dist)
    T = reduce(vcat, transpose.(fitm.embedding))
    Umap(fitm, T, xscales, s, par)
end

""" 
    transf(object::Umap, X)
Compute latent variables (LVs; = scores) from a fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
"""
function transf(object::Umap, X)
    res = UMAP.transform(object.fitm, Matrix(fscale(X, object.xscales)'))
    reduce(vcat, transpose.(res.embedding)) 
end

