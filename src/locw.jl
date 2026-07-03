"""
    locw(Xtrain::Matrix{Q}, Ytrain::Matrix{Q}, X::Matrix{Q}; listnn::Vector{Vector{Int}}, 
        listw::Union{Nothing, Vector{Vector{Q}}} = nothing, algo::Function, store::Bool = false, 
        verbose::Bool = true, kwargs...) where Q <: Float
    locw(Xtrain::Matrix{Q}, ytrain::Vector{String}, X::Matrix{Q}; listnn::Vector{Vector{Int}}, 
        listw::Union{Nothing, Vector{Vector{Q}}} = nothing, algo::Function, store::Bool = false, 
        verbose::Bool = true, kwargs...) where Q <: Float
Compute predictions for a given kNN model.
* `Xtrain` : Training X-data (n, p).
* `Ytrain` : Training Y-data (n, q).
* `ytrain` : Training Y-data (n) (class membership). Must be a `Vector{String}`.
* `X` : X-data (m, p) to predict.
Keyword arguments:
* `listnn` : List (vector) of m vectors of indexes.
* `listw` : List (vector) of m vectors of weights.
* `algo` : Function computing the model on the neighborhood of each new data to predict.
* `store` : Boolean. If `true`, the local models fitted on the neighborhoods are stored and returned 
    by function `predict` (output `fitm`).
* `verbose` : Boolean. If `true`, predicting information are printed.
* `kwargs` : Keywords arguments to pass in function `algo`. Each argument must have length = 1 (not be a collection).

Each component i of `listnn` and `listw` contains the indexes and weights, respectively, of the nearest neighbors 
of x_i in Xtrain. The sizes of the neighborhood for i = 1,...,m can be different.
"""
function locw(Xtrain::Matrix{Q}, Ytrain::Matrix{Q}, X::Matrix{Q}; listnn::Vector{Vector{Int}}, 
        listw::Union{Nothing, Vector{Vector{Q}}} = nothing, algo::Function, store::Bool = false, 
        verbose::Bool = true, kwargs...) where Q <: Float
    m = nro(X)
    q = nco(Ytrain)
    pred = similar(Ytrain, m, q)
    fitm = list(m)
    #@inbounds for i = 1:m
    Threads.@threads for i in eachindex(fitm)
        if verbose ; print(i, " ") ; end
        s = listnn[i]
        if length(s) == 1
            s = s:s
        end
        zXtrain = vrow(Xtrain, s)
        zYtrain = vrow(Ytrain, s)
        if isnothing(listw)
            zfitm = algo(zXtrain,  zYtrain; kwargs...)
        else
            zfitm = algo(zXtrain, zYtrain, pweight(listw[i]); kwargs...)
        end
        pred[i, :] = predict(zfitm, vrow(X, i:i)).pred
        if store ; fitm[i] = zfitm ; end 
    end
    if verbose ; println() ; end 
    (pred = pred, fitm)
end

function locw(Xtrain::Matrix{Q}, ytrain::Vector{String}, X::Matrix{Q}; listnn::Vector{Vector{Int}}, 
        listw::Union{Nothing, Vector{Vector{Q}}} = nothing, algo::Function, store::Bool = false, 
        verbose::Bool = true, kwargs...) where Q <: Float
    m = nro(X)
    pred = similar(Ytrain, m, 1)
    fitm = list(m)
    #@inbounds for i = 1:m
    Threads.@threads for i in eachindex(fitm)
        if verbose ; print(i, " ") ; end
        s = listnn[i]
        if length(s) == 1
            s = s:s
        end
        zXtrain = vrow(Xtrain, s)
        zytrain = vrow(ytrain, s)
        ## Case where all the neighbors have the same class
        if q == 1 && length(unique(zYtrain)) == 1
            pred[i, :] .= zytrain[1]
        ## End
        else
            if isnothing(listw)
                zfitm = algo(zXtrain,  zytrain; kwargs...)
            else
                zfitm = algo(zXtrain, zytrain, pweight(listw[i]); kwargs...)
            end
            pred[i, :] = predict(zfitm, vrow(X, i:i)).pred
            if store ; fitm[i] = zfitm ; end 
        end
    end
    if verbose ; println() ; end 
    (pred = pred, fitm)
end


