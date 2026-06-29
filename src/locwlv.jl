"""
    locwlv(Xtrain, Ytrain, X; listnn::Vector{Vector{Int}}, 
        listw::Union{Nothing, Vector{Vector{Q}}} = nothing, algo::Function, 
        nlv::Union{Int, AbstractVector{Int}}, 
        store::Bool = false, 
        verbose::Bool = true, kwargs...) where Q <: Float
* `Xtrain` : Training X-data.
* `Ytrain` : Training Y-data.
* `X` : X-data (m observations) to predict.
Keyword arguments:
* `listnn` : List (vector) of m vectors of indexes.
* `listw` : List (vector) of m vectors of weights.
* `algo` : Function computing the model on the m neighborhoods.
* `nlv` : Nb., or collection of nb., of latent variables (LVs).
* `store` : Boolean. If `true`, the local models fitted on the neighborhoods are stored and returned 
    by function `predict` (output `fitm`).
* `verbose` : Boolean. If `true`, predicting information are printed.
* `kwargs` : Keywords arguments to pass in function `algo`. Each argument must have length = 1 (not be a collection).

Same as [`locw`](@ref) but specific and much faster for LV-based models (e.g., PLSR).
"""
function locwlv(Xtrain, Ytrain, X; listnn::Vector{Vector{Int}}, 
        listw::Union{Nothing, Vector{Vector{Q}}} = nothing, algo::Function, 
        nlv::Union{Int, AbstractVector{Int}}, 
        store::Bool = false, 
        verbose::Bool = true, kwargs...) where Q <: Float
    p = nco(Xtrain)
    q = nco(Ytrain)
    m = nro(X)
    if isa(nlv, Int)
        nlv = min(nlv, p)
    else
        nlv = min(minimum(nlv), p):min(maximum(nlv), p)
    end
    le_nlv = length(nlv)
    zpred = similar(Ytrain, m, q, le_nlv)
    fitm = list(m)
    #@inbounds for i = 1:m
    Threads.@threads for i in eachindex(fitm)
        if verbose ; print(i, " ") ; end
        s = listnn[i]
        if length(s) == 1 ; s = s:s ; end
        zXtrain = vrow(Xtrain, s)
        zYtrain = Ytrain[s, :]   # vrow makes pb in aggsumv (e.g., lda) when Ytrain is a vector
        ## For discrimination, case where all the neighbors have the same class
        if q == 1 && length(unique(zYtrain)) == 1
            @inbounds for a in eachindex(nlv)
                zpred[i, :, a] .= zYtrain[1]
            end
        ## End 
        else
            if isnothing(listw)
                zfitm = algo(zXtrain,  zYtrain; nlv = maximum(nlv), kwargs...)
            else
                zfitm = algo(zXtrain, zYtrain, pweight(listw[i]); nlv = maximum(nlv), kwargs...)
            end
            vpred = predict(zfitm, vrow(X, i:i), nlv).pred
            @inbounds for a in eachindex(nlv)
                zpred[i, :, a] = vpred[a]
            end
            if store ; fitm[i] = zfitm ; end 
        end
    end 
    if verbose ; println() ; end    
    pred = list(Matrix, le_nlv)
    @inbounds for a in eachindex(nlv)
        pred[a] = zpred[:, :, a]
    end
    (pred = pred, fitm)
end
