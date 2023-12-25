"""
    locwlv(Xtrain, Ytrain, X; 
        listnn, listw = nothing, fun, nlv, 
        verbose = true, kwargs...)
Compute predictions for a given kNN model.
* `Xtrain` : Training X-data.
* `Ytrain` : Training Y-data.
* `X` : X-data (m observations) to predict.
Keyword arguments:
* `listnn` : List (vector) of m vectors of indexes.
* `listw` : List (vector) of m vectors of weights.
* `fun` : Function computing the model on 
    the m neighborhoods.
* `nlv` : Nb. or collection of nb. of latent variables (LVs).
* `verbose` : Boolean. If true, fitting information 
    are printed.
* `kwargs` : Keywords arguments to pass in function fun. 

Same as [`locw`](@ref) but specific and much faster 
for LV-based models (e.g. PLSR).
"""
function locwlv(Xtrain, Ytrain, X; 
        listnn, listw = nothing, fun, nlv, 
        verbose = true, kwargs...)
    p = nco(Xtrain)
    m = nro(X)
    q = nco(Ytrain)
    nlv = max(0, minimum(nlv)):min(p, maximum(nlv))
    le_nlv = length(nlv)
    zpred = similar(Ytrain, m, q, le_nlv)
    Threads.@threads for i = 1:m
    #@inbounds for i = 1:m
        verbose ? print(i, " ") : nothing
        s = listnn[i]
        length(s) == 1 ? s = (s:s) : nothing
        zYtrain = Ytrain[s, :]
        ## For discrimination,
        ## case where all the neighbors are of same class
        if q == 1 && length(unique(zYtrain)) == 1
            @inbounds for a = 1:le_nlv
                zpred[i, :, a] .= zYtrain[1]
            end
        ## End 
        else
            if isnothing(listw)
                fm = fun(Xtrain[s, :],  zYtrain; 
                    nlv = maximum(nlv), kwargs...)
            else
                fm = fun(Xtrain[s, :], zYtrain, mweight(listw[i]); 
                    nlv = maximum(nlv), kwargs...)
            end
            @inbounds for a = 1:le_nlv
                zpred[i, :, a] = predict(fm, X[i:i, :]; 
                    nlv = nlv[a]).pred
            end
        end
    end 
    verbose ? println() : nothing    
    pred = list(Union{Matrix{Int}, Matrix, Matrix},
        le_nlv)
    for a = 1:le_nlv
        pred[a] = zpred[:, :, a]
    end
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred, )
end

