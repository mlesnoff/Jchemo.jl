"""
    locw(Xtrain, Ytrain, X ; listnn, listw = nothing, fun, verbose = false, kwargs...)
Compute a prediction score (error rate; e.g. RMSEP) for a given model over a grid of parameter values.
- Xtrain : Training X-data, (n, p) or (n,).
- Ytrain : Training Y-data, (n, q) or (n,).
- X : Query X-data, (m, p) or (m,).
- listnn : List (m,) of indexes of the nearest neighbors of X in Xtrain.
- listw : List (m,) of the weights of the nearest neighbors of X in Xtrain
- fun : Function computing the model on the m neighborhoods.
- verbose : If true, fitting information are printed.
- kwargs : Keywords arguments (parameters) to pass in function fun. 

Set only 1 combination of parameters in kwargs,
i.e. all arguments in kwargs must have length = 1 (not collections).
"""
function locw(Xtrain, Ytrain, X ; 
    listnn, listw = nothing, fun, verbose = false, kwargs...)
    m = size(X, 1)
    q = size(Ytrain, 2)
    pred = zeros(m, q)
    for i = 1:m
        verbose ? print(i, " ") : nothing
        s = listnn[i]
        length(s) == 1 ? s = (s:s) : nothing
        zYtrain = Ytrain[s, :]
        if q == 1 & length(unique(zYtrain)) == 1
            ## For discrimination, 
            ## case where all the neighbors are of same class
            pred[i, :] .= zYtrain[1]
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
    locwlv(Xtrain, Ytrain, X ; listnn, listw = nothing, fun, nlv, verbose = false, kwargs...)
Same as [`locw`](@ref) but specific (and much faster) to LV-based models (e.g. PLSR).
- nlv : Nb. or collection of nb. of latent variables (LVs).

Argument pars must not contain nlv.
"""
function locwlv(Xtrain, Ytrain, X ; 
    listnn, listw = nothing, fun, nlv, verbose = true, kwargs...)
    m = size(X, 1)
    q = size(Ytrain, 2)
    nlv = max(minimum(nlv), 0):maximum(nlv)
    le_nlv = length(nlv)
    zpred = zeros(m, q, le_nlv)
    #zpred = Array{Float64}(undef, m, q, le_nlv)
    for i = 1:m
        verbose ? print(i, " ") : nothing
        s = listnn[i]
        length(s) == 1 ? s = (s:s) : nothing
        zYtrain = Ytrain[s, :]
        ## For discrimination,
        ## case where all the neighbors are of same class
        if q == 1 & length(unique(zYtrain)) == 1
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
            for a = 1:le_nlv
                zpred[i, :, a] = predict(fm, X[i:i, :] ; nlv = nlv[a]).pred
            end
        end
    end 
    verbose ? println() : nothing    
    pred = list(le_nlv)
    for a = 1:le_nlv
        pred[a] = zpred[:, :, a]
    end
    le_nlv == 1 ? pred = pred[1] : nothing
    (pred = pred, )
end





