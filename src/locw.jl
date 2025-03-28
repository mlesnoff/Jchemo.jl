"""
    locw(Xtrain, Ytrain, X; listnn, listw = nothing, algo, verbose = false, kwargs...)
Compute predictions for a given kNN model.
* `Xtrain` : Training X-data.
* `Ytrain` : Training Y-data.
* `X` : X-data (m observations) to predict.
Keyword arguments:
* `listnn` : List (vector) of m vectors of indexes.
* `listw` : List (vector) of m vectors of weights.
* `algo` : Function computing the model on 
    the m neighborhoods.
* `verbose` : Boolean. If `true`, predicting information are printed.
* `kwargs` : Keywords arguments to pass in function `algo`.
    Each argument must have length = 1 (not be a collection).

Each component i of `listnn` and `listw` contains the indexes 
and weights, respectively, of the nearest neighbors of x_i in Xtrain. 
The sizes of the neighborhood for i = 1,...,m can be different.
"""
function locw(Xtrain, Ytrain, X; listnn, listw = nothing, algo, verbose = false, kwargs...)
    m = nro(X)
    q = nco(Ytrain)
    pred = similar(Ytrain, m, q)
    #@inbounds for i = 1:m
    Threads.@threads for i = 1:m
        verbose ? print(i, " ") : nothing
        s = listnn[i]
        length(s) == 1 ? s = (s:s) : nothing
        zXtrain = vrow(Xtrain, s)
        zYtrain = Ytrain[s, :]   # vrow makes pb in aggsumv (e.g. lda) when Ytrain is a vector
        ## For discrimination, 
        ## case where all the neighbors have the same class
        if q == 1 && length(unique(zYtrain)) == 1
            pred[i, :] .= zYtrain[1]
        ## End
        else
            if isnothing(listw)
                fitm = algo(zXtrain,  zYtrain; kwargs...)
            else
                fitm = algo(zXtrain, zYtrain, mweight(listw[i]); kwargs...)
            end
            pred[i, :] = predict(fitm, vrow(X, i:i)).pred
        end
    end
    verbose ? println() : nothing    
    (pred = pred,)
end


