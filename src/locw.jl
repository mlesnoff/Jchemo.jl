"""
    locw(Xtrain, Ytrain, X; listnn, listw = nothing, algo, verbose = false, kwargs...)
Compute predictions for a given kNN model.
* `Xtrain` : Training X-data.
* `Ytrain` : Training Y-data.
* `X` : X-data (m observations) to predict.
Keyword arguments:
* `listnn` : List (vector) of m vectors of indexes.
* `listw` : List (vector) of m vectors of weights.
* `algo` : Function computing the model on the m neighborhoods.
* `store` : Boolean. If `true`, the local models fitted on the neighborhoods are stored and returned by function `predict`.
* `verbose` : Boolean. If `true`, predicting information are printed.
* `kwargs` : Keywords arguments to pass in function `algo`. Each argument must have length = 1 (not be a collection).

Each component i of `listnn` and `listw` contains the indexes and weights, respectively, of the nearest neighbors 
of x_i in Xtrain. The sizes of the neighborhood for i = 1,...,m can be different.
"""
function locw(Xtrain, Ytrain, X; listnn, listw = nothing, algo, store = false, verbose = false, kwargs...)
    m = nro(X)
    q = nco(Ytrain)
    pred = similar(Ytrain, m, q)
    fitm = list(m)
    #@inbounds for i = 1:m
    Threads.@threads for i = 1:m
        verbose ? print(i, " ") : nothing
        s = listnn[i]
        length(s) == 1 ? s = (s:s) : nothing
        zXtrain = vrow(Xtrain, s)
        zYtrain = Ytrain[s, :]   # vrow makes pb in aggsumv (e.g., lda) when Ytrain is a vector
        ## For discrimination, 
        ## case where all the neighbors have the same class
        if q == 1 && length(unique(zYtrain)) == 1
            fitm[i] = nothing
            pred[i, :] .= zYtrain[1]
        ## End
        else
            if isnothing(listw)
                fitm[i] = algo(zXtrain,  zYtrain; kwargs...)
            else
                fitm[i] = algo(zXtrain, zYtrain, mweight(listw[i]); kwargs...)
            end
            pred[i, :] = predict(fitm[i], vrow(X, i:i)).pred
        end
    end
    if !store ; fitm = nothing ; end
    verbose ? println() : nothing    
    (pred = pred, fitm)
end


