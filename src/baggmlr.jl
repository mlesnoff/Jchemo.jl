struct Baggmlr1
    fm
end

""" 
    baggmlr(X, Y, weights = nothing ; B, fun = mlrchol,
        rowsamp = .7, withr = false, 
        colsamp = .3)
Bagging of linear model (LMs) regressions.
* `X` : X-data (n obs., p variables).
* `Y` : Y-data (n obs., q variables).
* `weights` : Weights of the observations.
* `B` : Nb. of bagging repetitions.
* `fun` : Function computing the LRM model.
* `withr`: Type of sampling of the observations
    (`true` => with replacement).
* `rowsamp` : Proportion of rows sampled in `X` 
    at each repetition.
* `colsamp` : Proportion of columns sampled (without replacement) in `X` 
    at each repetition.


""" 
function baggmlr(X, Y, weights = nothing ; B, fun = mlrchol,
    rowsamp = .7, withr = false, colsamp = .3)
        
    fm = baggr(X, Y, weights; B = B, fun = fun, 
        rowsamp = rowsamp, withr = withr, colsamp = colsamp) ;

    Baggmlr1(fm)
end

function predict(object::Baggmlr1, X)
    pred = Jchemo.predict(object.fm, X).pred ;    
    (pred = pred,)
end




