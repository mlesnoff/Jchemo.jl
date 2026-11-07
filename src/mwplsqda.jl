"""
    mwplsqda(; kwargs...)
    mwplsqda(X, y, group::Union{Nothing, Vector{String}} = nothing; kwargs...)
    mwplsqda(X::AbstractMatrix{Q}, y::AbstractVector{String}, weights::ProbabilityWeights{Q},
        group::Union{Nothing, Vector{String}}; kwargs...) where Q <: Float
Moving window PLS-QDA (MWPLS-QDA).
* `X` : X-data (n, p).
* `y` : Univariate class membership (n). Must be a `Vector{String}`.
* `group` : A vector (n) defining groups (batches) of observations. Must be a `Vector{String}`.
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `npoint` : Total number of points in the sliding window (must be odd).
* `K` : Number of folds in the K-fold cross validation used to select the nb. LVs for each window. 
* `rep` : Number of repetitions of the K-fold cross validation (CV) used to select the nb. LVs for each window.
* `h` : A scalar defining the shape of the weight function computed by function `winvs`. Lower is h, 
    sharper is the function. See function `winvs` for details. 
* `score` : Function computing the error rate (to minimize) in the CV (by default, `score = errp`).
* `nlv` : Maximum nb. of latent variables (LVs) to consider in each PLS model.
* `prior` : Type of prior probabilities for class membership. Possible values are: `:prop` (proportionnal), 
    `:unif` (uniform), or a vector (of length equal to the number of classes) giving the prior weight for each class 
    (in case of vector, it must be sorted in the same order as `mlev(y)`).
* `alpha` : Scalar (∈ [0, 1]) defining the continuum between QDA (`alpha = 0`) and LDA (`alpha = 1`).
* `scal` : Symbol defining the column scaling of `X` and Ydummy. Possible values are: `:none` (default), 
    `:std` (uncorrected STD), `:prt` (pareto) and `:mad` (MAD).

The function implements a moving (i.e., sliding) window PLS discrimination by QDA.

See function `mwplsr` for details. The principle is the same except that a PLS-QDA is run instead
of a PLSR. 

# Examples
```julia
```
"""
mwplsqda(; kwargs...) = JchemoModel(mwplsqda, nothing, kwargs)

function mwplsqda(X, y, group::Union{Nothing, Vector{String}} = nothing; kwargs...)
    X = ensure_mat(X)
    y = vec(y)
    weights = pweight(ones(eltype(X), nro(X)))
    mwplsqda(X, y, weights, group; kwargs...)
end

function mwplsqda(X::AbstractMatrix{Q}, y::AbstractVector{String}, weights::ProbabilityWeights{Q},
        group::Union{Nothing, Vector{String}}; kwargs...) where Q <: Float
    n, p = size(X)       
    par = recovkw(ParMwplsqda{Q}, kwargs).par
    @assert isodd(par.npoint) && par.npoint >= 1 "Argument 'npoint' must be an odd integer >= 1."
    nhwindow = Int((par.npoint - 1) / 2)  # half window
    rangetot = (nhwindow + 1):(p - nhwindow)
    centrw = collect(rangetot)
    nw = length(rangetot)
    window = list(UnitRange{Int}, nw)
    fitm_emb = list(Plsprobda, nw)
    nlv_emb = list(Int, nw)
    scor_emb = list(Q, nw)
    j = 1   
    @inbounds for i in rangetot                     # run for each window 'j'
        window[j] = (i - nhwindow):(i + nhwindow)   # window 'j'
        vX = vcol(X, window[j])
        # Cross-validate (only nlv) to select the model
        if isnothing(group) 
            segm = segmkf(n, par.K; rep = par.rep)
        else
            segm = segmkf(group, par.K; rep = par.rep)
        end
        pars = mpar(alpha = par.alpha, prior = [par.prior], scal = [par.scal])
        model = plsqda()
        res = gridcv(model, vX, y; segm, score = par.score, nlv = 1:par.nlv, pars).res
        u = first(findall(res.y1 .== minimum(res.y1))) 
        scor_emb[j] = res.y1[u]
        nlv_emb[j] = res.nlv[u]
        # Fit the selected model        
        fitm_emb[j] = plsqda(vX, y, weights; nlv = nlv_emb[j], alpha = par.alpha, 
            prior = par.prior, scal = par.scal)
        j = j + 1
    end
    # Model weights
    scor_best = minimum(scor_emb)
    d_emb = scor_emb .- scor_best   # model performance discrepancy
    criw = 3. ; squared = false
    w_emb = winvs(d_emb; h = par.h, criw, squared)
    wtot = sum(w_emb)
    @. w_emb /= wtot                # sum(w_emb) = 1
    # Variable importance 
    V = -ones(Q, nw, p)
    @inbounds for i in eachindex(fitm_emb)
        V[i, window[i]] .= w_emb[i]
    end
    vi = similar(X, p)
    @inbounds for j in axes(V, 2)
        v = vcol(V, j)
        vi[j] = meanv(v[v .> -1])
    end
    Mwplsqda(fitm_emb, nlv_emb, scor_emb, d_emb, w_emb, vi, window, centrw) 
end

function predict(object::Mwplsqda, X)
    X = ensure_mat(X)
    Q = eltype(X)
    m = nro(X)
    nw = length(object.fitm_emb)
    predw = similar(X, m, 1, nw)
    pred = zeros(Q, m, q)
    # Computing votes
    @inbounds for i in eachindex(object.fitm_emb)
        predw[:, :, i] = predict(object.fitm_emb[i], vcol(X, object.window[i])).pred
        pred = pred + object.w_emb[i] * predw[:, :, i] 
    end
    (pred = pred, predw)
end


