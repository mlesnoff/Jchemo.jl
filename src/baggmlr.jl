struct Baggmlr
    fm
    B
    int
end

""" 
    baggmlr(X, Y, weights = nothing ; fun = mlrchol, rep, 
        rowsamp = .7, withr = false, 
        colsamp = .3)
Bagging of linear model (LMs) regressions.
* `X` : X-data (n obs., p variables).
* `Y` : Y-data (n obs., q variables).
* `weights` : Weights of the observations.
* `rep` : Nb. of bagging repetitions.
* `fun` : Function computing the LRM model.
* `withr`: Type of sampling of the observations
    (`true` => with replacement).
* `rowsamp` : Proportion of rows sampled in `X` 
    at each repetition.
* `colsamp` : Proportion of columns sampled (without replacement) in `X` 
    at each repetition.

""" 
function baggmlr(X, Y, weights = nothing ; fun = mlrchol, rep, 
    rowsamp = .7, withr = false, colsamp = .3)
    p = size(X, 2)
    q = size(Y, 2)  
    fm = baggr(X, Y, weights; fun = fun, rep = rep, 
        rowsamp = rowsamp, withr = withr, colsamp = colsamp) ;
    z = zeros(p, q, rep)
    for i = 1:rep
        u = fm.s_col[:, i]
        z[u, :, i] .= fm.fm[i].B
    end
    B = mean(z, dims = 3)[:, :, 1]
    #z = zeros(1, q, rep)
    for i = 1:rep
        z[1, :, i] .= vec(fm.fm[i].int)
    end
    int = mean(z[1:1, :, :], dims = 3)[:, :, 1]
    Baggmlr(fm, B, int)
end

function predict_baggmlr(object, X)
    pred = Jchemo.predict(object.fm, X).pred ;    
    (pred = pred,)
end




