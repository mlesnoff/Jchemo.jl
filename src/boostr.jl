struct Boostr
    fm
    s_row
    s_col
    beta
    probs
end

""" 
    boostr(X, Y, weights = nothing; B, fun, 
        rowsamp = 1, withr = false, colsamp = 1, meth = "dru", kwargs...)
Adaptative boosting (sampling) for regression models.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `B` : Nb. of boosting iterations.
* `fun` : Name (string) of the function computing the model to boost.
* `rowsamp` : Proportion of rows (observations) sampled in `X` at each iteration.
* `withr`: Boolean defining the type of sampling of the observations when `rowsamp` < 1 
    (`withr = false` => sampling without replacement).
* `colsamp` : Proportion of columns (variables) sampled in `X` at each iteration.
* `kwargs` : Optional named arguments to pass in 'fun`.

This is the AdaBoost algorithm of Drucker 1997,
which is an adaptation of the AdaBoost.M1 classificatuon algorithm 
of Freund & Schapire 1997.

Assume that `X` is (n, p).

If `rowsamp` = 1, each boosting iteration is run on all the n observations 
(no preliminary sampling).

If `rowsamp` < 1, each boosting iteration is done on `rowsamp` * n sampled 
observations. The sampling can be without (default) or with replacement, 
depending on argument `withr`.

If `colsamp` < 1 , a proportion of `colsamp` * p variables are sampled without replacement 
at each boosting iteration, and taken as predictors for the given iteration.

## References

Drucker, H., 1997. Improving regressor using boosting techniques, 
in: Proc. of the 14th International Conferences on Machine Learning. 
Morgan Kaufmann, San Mateo, CA, pp. 107–115.

Freund, Y., Schapire, R.E., 1997. A Decision-Theoretic Generalization of 
On-Line Learning and an Application to Boosting. Journal of Computer and System Sciences 55, 
119–139. https://doi.org/10.1006/jcss.1997.1504

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245

Shrestha, D.L., Solomatine, D.P., 2006. Experiments with AdaBoost.RT, 
an Improved Boosting Scheme for Regression. Neural Computation 18, 1678–1710.
https://doi.org/10.1162/neco.2006.18.7.1678

Solomatine, D., Shrestha, D., 2004. AdaBoost.RT: A boosting algorithm for regression problems. 
Presented at the IEEE International Conference on Neural Networks
- Conference Proceedings, pp. 1163–1168 vol.2. https://doi.org/10.1109/IJCNN.2004.1380102

""" 
function boostr(X, Y, weights = nothing; fun, B, 
    withr = false, rowsamp = 1, colsamp = 1, kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    q = size(Y, 2)
    flearn = eval(Meta.parse(fun))    
    fm = list(B)
    n_row = Int64(round(rowsamp * n))
    n_col = max(Int64(round(colsamp * p)), 1)
    s_row = fill(1, (n_row, B))
    s_col = similar(s_row, n_col, B) 
    srow = similar(s_row, n_row)
    scol = similar(s_row, n_col)
    probs = similar(X, n, B) # weights of the observations in the sampling
    zprobs = ones(n) / n
    r2 = similar(X, n)
    d = similar(X, n)
    beta = similar(X, B)
    w = similar(X, n_row)
    zncol = collect(1:n_col) 
    zX = similar(X, n_row, n_col)
    zY = similar(Y, n_row, q)
    @inbounds for i = 1:B
        srow .= sample(1:n, Weights(zprobs), n_row; replace = withr)
        if colsamp == 1
            scol .= zncol
        else
            scol .= sample(1:p, n_col; replace = false) 
        end
        zX .= X[srow, scol]
        zY .= Y[srow, :]       
        if(isnothing(weights))
            fm[i] = flearn(zX, zY; kwargs...)
        else
            w .= mweights(weights[srow])
            fm[i] = flearn(zX, zY, w; kwargs...)
        end
        # @view is not accepted by XGBoost.predict
        pred = predict(fm[i], X[:, scol]).pred
        r2 .= vec(sum(residreg(pred, Y).^2, dims = 2))
        d .= r2 / maximum(r2)[1]               
        L = dot(zprobs, d)
        beta[i] = L / (1 - L)
        if L <= .5                                  # or beta[i] >= 1
            zprobs .= mweights(beta[i].^(1 .- d))
        else
            zprobs .= ones(n) / n
        end
        s_row[:, i] .= srow
        s_col[:, i] .= scol
        probs[:, i] .= zprobs
    end
    Boostr(fm, s_row, s_col, beta, probs)
end

function predict(object::Boostr, X)
    B = length(object.fm)
    scol = vcol(object.s_col, 1)
    w = log.(1 ./ object.beta)
    w .= mweights(w)
    # @view is not accepted by XGBoost.predict  
    acc = w[1] * predict(object.fm[1], X[:, scol]).pred
    @inbounds for i = 2:B
        scol = vcol(object.s_col, i)
        acc .+= w[i] * predict(object.fm[i], X[:, scol]).pred
    end
    (pred = acc,)
end

################ Direct weighting

""" 
    boostrw(X, Y, weights = nothing; B, fun, 
        rowsamp = 1, withr = false, colsamp = 1, kwargs...)
Adaptative boosting (direct) for regression models.

Same as `boostr` except that the boosting weights computed for the 
n observations are directly accounted for in the fitting process 
(there is no sampling of observations).
""" 
function boostrw(X, Y; fun, B, 
    withr = false, rowsamp = 1, colsamp = 1, kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    q = size(Y, 2)
    flearn = eval(Meta.parse(fun))    
    fm = list(B)
    n_row = Int64(round(rowsamp * n))
    n_col = max(Int64(round(colsamp * p)), 1)
    s_row = fill(1, (n_row, B))
    s_col = similar(s_row, n_col, B) 
    srow = similar(s_row, n_row)
    scol = similar(s_row, n_col)
    probs = similar(X, n, B)  # weights of the observations in the fitting
    zprobs = ones(n) / n
    r2 = similar(X, n)
    d = similar(X, n)
    beta = similar(X, B)
    zn = collect(1:n)
    zncol = collect(1:n_col) 
    zX = similar(X, n_row, n_col)
    zY = similar(Y, n_row, q)
    @inbounds for i = 1:B
        if rowsamp == 1
            srow .= zn
        else
            srow .= sample(1:n, n_row; replace = withr)
        end
        if colsamp == 1
            scol .= zncol
        else
            scol .= sample(1:p, n_col; replace = false) 
        end
        zX .= @view(X[srow, scol])
        zY .= @view(Y[srow, :])       
        fm[i] = flearn(zX, zY, zprobs[srow]; kwargs...)
        # @view is not accepted by XGBoost.predict
        # @view(X[:, scol])
        pred = predict(fm[i], X[:, scol]).pred
        r2 .= vec(sum(residreg(pred, Y).^2, dims = 2))
        d .= r2 / maximum(r2)[1]               
        L = dot(zprobs, d)
        beta[i] = L / (1 - L)
        if L <= .5                                    
            zprobs .= mweights(beta[i].^(1 .- d))
        else
            zprobs .= ones(n) / n
        end    
        s_row[:, i] .= srow
        s_col[:, i] .= scol
        probs[:, i] .= zprobs
    end
    Boostr(fm, s_row, s_col, beta, probs)
end


