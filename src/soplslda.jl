"""
    soplslda(; kwargs...)
    soplslda(Xbl, y; kwargs...)
    soplslda(Xbl::Vector{Matrix{Q}}, y::Vector{String}, weights::ProbabilityWeights{Q}; 
        kwargs...) where Q <: Float
LDA on SOPLS latent variables (SOPLS-LDA).
* `Xbl` : List of blocks (vector of matrices) of X-data. Typically, output of function `mblock` from data (n, p).  
* `y` : Univariate class membership (n). Must be a `Vector{String}`.
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs; = scores) to compute. Can be a single integer (same nb. LVs for each block) 
    or a vector of integers (nb. LVs for each block).
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `scal` : Symbol defining the column scaling of blocks in `Xbl` and Ydummy (before the block scaling). 
    Possible values are: `:none` (default), `:std` (uncorrected STD), `:prt` (pareto) and `:mad` (MAD).

Same as PLS-LDA but LDA is run on SOPLSR scores (computed using the class dummy table, Ydummy).

# Examples
```julia
```

# References
Biancolillo, A., 2016. Method development in the area of multi-block analysis focused on food analysis. 
PhD. University of Copenhagen.

Biancolillo, A., Måge, I., Næs, T., 2015. Combining SO-PLS and linear discriminant analysis 
for multi-block classification. Chemometrics and Intelligent Laboratory Systems 141, 58–67. 
https://doi.org/10.1016/j.chemolab.2014.12.001

Menichelli et al., 2014. SO-PLS as an exploratory tool for path modelling. 
Food Quality and Preference, 36, 122-134.
"""
soplslda(; kwargs...) = JchemoModel(soplslda, nothing, kwargs)

function soplslda(Xbl, y; kwargs...)
    Xbl = ensure_mat_mb(Xbl)
    y = vec(y)
    Q = eltype(Xbl[1])
    prior = recovkw(ParSoplsda{Q}, kwargs).par.prior
    weights = pweightcla(Q, y; prior)
    Soplsprobda(Xbl, y, weights; kwargs...)
end

function soplslda(Xbl::Vector{Matrix{Q}}, y::Vector{String}, weights::ProbabilityWeights{Q}; 
        kwargs...) where Q <: Float
    par = recovkw(ParSoplsda{Q}, kwargs).par
    res = dummy(Q, y)
    ni = tab(y).n
    priors = aggsumv(weights.values, y).val  # output not used, only for information
    fitm_emb = soplsr(Xbl, res.Y, weights; kwargs...)
    par.nlv = fitm_emb.par.nlv
    fitm_da = lda(fitm_emb.T, y, weights; kwargs...)
    Soplsprobda(fitm_emb, fitm_da, ni, priors, res.lev, par) 
end

"""
    predict(object::Soplsprobda, Xbl)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `Xbl` : A list of blocks (vector of matrices) of X-data for which predictions are computed.
""" 
function predict(object::Soplsprobda, Xbl)
    T = transf(object.fitm_emb, Xbl)
    res = predict(object.fitm_da, T)
    (pred = res.pred, posterior = res.posterior, nlv = nco(T))
end


