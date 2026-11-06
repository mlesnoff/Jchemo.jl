"""
    soplsqda(; kwargs...)
    soplsqda(Xbl, y; kwargs...)
    soplsqda(Xbl::Vector{Matrix{Q}}, y::Vector{String}, weights::ProbabilityWeights{Q}; 
        kwargs...) where Q <: Float
QDA on SOPLS latent variables (SOPLS-QDA).
* `Xbl` : List of blocks (vector of matrices) of X-data. Typically, output of function `mblock` from data (n, p).  
* `y` : Univariate class membership (n). Must be a `Vector{String}`.
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs; = scores) to compute. Can be a single integer (same nb. LVs for each block) 
    or a vector of integers (nb. LVs for each block).
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum between QDA (`alpha = 0`) and LDA (`alpha = 1`).
* `scal` : Symbol defining the column scaling of blocks in `Xbl` and Ydummy (before the block scaling). 
    Possible values are: `:none` (default), `:std` (uncorrected STD), `:prt` (pareto) and `:mad` (MAD).

Same as PLS-QDA but QDA is run on SOPLSR scores (computed using the class dummy table, Ydummy).

See function `soplslda` for examples.

# References
Biancolillo, A., 2016. Method development in the area of multi-block analysis focused on food analysis. 
PhD. University of Copenhagen.

Biancolillo, A., Måge, I., Næs, T., 2015. Combining SO-PLS and linear discriminant analysis 
for multi-block classification. Chemometrics and Intelligent Laboratory Systems 141, 58–67. 
https://doi.org/10.1016/j.chemolab.2014.12.001

Menichelli et al., 2014. SO-PLS as an exploratory tool for path modelling. 
Food Quality and Preference, 36, 122-134.
"""
soplsqda(; kwargs...) = JchemoModel(soplsqda, nothing, kwargs)

function soplsqda(Xbl, y; kwargs...)
    Xbl = ensure_mat_mb(Xbl)
    y = vec(y)
    Q = eltype(Xbl[1])
    prior = recovkw(ParSoplsqda{Q}, kwargs).par.prior
    weights = pweightcla(Q, y; prior)
    soplsqda(Xbl, y, weights; kwargs...)
end

function soplsqda(Xbl::Vector{Matrix{Q}}, y::Vector{String}, weights::ProbabilityWeights{Q}; 
        kwargs...) where Q <: Float
    par = recovkw(ParSoplsqda{Q}, kwargs).par
    res = dummy(Q, y)
    ni = tab(y).n
    priors = aggsumv(weights.values, y).val  # output not used, only for information
    fitm_emb = soplsr(Xbl, res.Y, weights; kwargs...)
    par.nlv = fitm_emb.par.nlv
    fitm_da = qda(fitm_emb.T, y, weights; kwargs...)
    Soplsprobda(fitm_emb, fitm_da, ni, priors, res.lev, par) 
end



