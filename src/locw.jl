"""
    locw(Xtrain, Ytrain, X ; 
        listnn, listw = nothing, fun, verbose = false, kwargs...)
Compute predictions for a given kNN model.
* `Xtrain` : Training X-data.
* `Ytrain` : Training Y-data.
* `X` : X-data (m observations) to predict.
* `listnn` : List of m vectors of indexes.
* `listw` : List of m vectors of weights.
* `fun` : Function computing the model on the m neighborhoods.
* `verbose` : If true, fitting information are printed.
* `kwargs` : Keywords arguments to pass in function fun. 

Each component i of `listnn` and `listw` contains the indexes and weights, respectively,
of the nearest neighbors of x_i in Xtrain. The sizes of the neighborhood 
for i = 1,...,m can be different.

All the arguments in kwargs must have length = 1 (not collections).
"""
function locw(Xtrain, Ytrain, X ; 
    listnn, listw = nothing, fun, verbose = false, kwargs...)
    m = size(X, 1)
    q = size(Ytrain, 2)
    #pred = zeros(m, q)
    pred = similar(Ytrain, m, q)
    @inbounds for i = 1:m
        verbose ? print(i, " ") : nothing
        s = listnn[i]
        length(s) == 1 ? s = (s:s) : nothing
        zYtrain = Ytrain[s, :]
        ## For discrimination, 
        ## case where all the neighbors are of same class
        if q == 1 && length(unique(zYtrain)) == 1
            pred[i, :] .= zYtrain[1]
        ## End
        else
            if isnothing(listw)
                fm = fun(Xtrain[s, :],  zYtrain; kwargs...)
            else
                fm = fun(Xtrain[s, :], zYtrain, listw[i] ; kwargs...)
            end
            pred[i, :] = predict(fm, X[i:i, :]).pred
        end
    end
    verbose ? println() : nothing    
    (pred = pred,)
end

"""
    locwlv(Xtrain, Ytrain, X; 
        listnn, listw = nothing, fun, nlv, verbose = true, kwargs...)
Compute predictions for a given kNN model.
* `nlv` : Nb. or collection of nb. of latent variables (LVs).

Same as [`locw`](@ref) but specific (and much faster) for LV-based (e.g. PLSR) models.
"""
function locwlv(Xtrain, Ytrain, X; 
    listnn, listw = nothing, fun, nlv, verbose = true, kwargs...)
    p = nco(Xtrain)
    m = nro(X)
    q = nco(Ytrain)
    nlv = max(0, minimum(nlv)):min(p, maximum(nlv))
    le_nlv = length(nlv)
    zpred = similar(Ytrain, m, q, le_nlv)
    @inbounds for i = 1:m
        verbose ? print(i, " ") : nothing
        s = listnn[i]
        length(s) == 1 ? s = (s:s) : nothing
        zYtrain = Ytrain[s, :]
        ## For discrimination,
        ## case where all the neighbors are of same class
        if q == 1 && length(unique(zYtrain)) == 1
            for a = 1:le_nlv
                zpred[i, :, a] .= zYtrain[1]
            end
        ## End 
        else
            if isnothing(listw)
                fm = fun(Xtrain[s, :],  zYtrain ; nlv = maximum(nlv), kwargs...)
            else
                fm = fun(Xtrain[s, :], zYtrain, listw[i] ; nlv = maximum(nlv), kwargs...)
            end
            @inbounds for a = 1:le_nlv
                zpred[i, :, a] = predict(fm, X[i:i, :] ; nlv = nlv[a]).pred
            end
        end
    end 
    verbose ? println() : nothing    
    pred = list(le_nlv, Union{Matrix{Int64}, Matrix{Float64}, Matrix{String}})
    for a = 1:le_nlv
        pred[a] = zpred[:, :, a]
    end
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred, )
end





