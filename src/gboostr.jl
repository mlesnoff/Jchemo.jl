struct Gboostr
    fm
    s_row
    s_col
    r_array
    nu
end

""" 
    gboostr(X, Y, weights = nothing; fun, B = 1, nu = 1, 
        samp_row = 1, withr = false, samp_col = 1, kwargs...)
Gradient boosting for regression models.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `fun` : Name (string) of the function computing the model to boost.
* `B` : Nb. of boosting iterations.
* `nu` : Learning rate (must be between 0 and 1) applied to each iteration
    (such as defined in Friedman Ann. Stat. 2001, Eq.36)
* `samp_row` : Proportion of rows (observations) sampled in `X` at each iteration.
* `withr`: Boolean defining the type of sampling of the observations when `samp_row` < 1 
    (`withr = false` => sampling without replacement).
* `samp_col` : Proportion of columns (variables) sampled in `X` at each iteration.
* `kwargs` : Optional named arguments to pass in 'fun`.

Assume that `X` is (n, p).

If `samp_row` = 1, each boosting iteration takes all the observations 
(no preliminary sampling).

If `samp_row` < 1, each boosting iteration is done on `samp_row` * n sampled observations, 
which corresponds to the stochastic gradient boosting. 
The sampling can be without (default) or with replacement, 
depending on argument `withr`.

If `samp_col` < 1 , a proportion of `samp_col` * p variables are sampled without replacement 
at each boosting iteration, and taken as predictors for the given iteration.

## References

Breiman, L., 2001. Using Iterated Bagging to Debias Regressions. 
Machine Learning 45, 261–277. https://doi.org/10.1023/A:1017934522171

Friedman, J.H., 2001. Greedy Function Approximation: A Gradient Boosting Machine. 
The Annals of Statistics 29, 1189–1232.

Friedman, J.H., 2002. Stochastic gradient boosting. 
Computational Statistics & Data Analysis, Nonlinear Methods and Data Mining 38, 367–378.
https://doi.org/10.1016/S0167-9473(01)00065-2

""" 
function gboostr(X, Y, weights = nothing; fun, B = 1, nu = 1, 
    samp_row = 1, withr = false, samp_col = 1, kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    q = size(Y, 2)
    flearn = eval(Meta.parse(fun))    
    fm = list(B)
    n_row = Int64(round(samp_row * n))
    n_col = max(Int64(round(samp_col * p)), 1)
    r_array = similar(X, n, q, B + 1) # residuals    
    s_row = fill(1, (n_row, B))
    s_col = similar(s_row, n_col, B)
    srow = similar(s_row, n_row)
    scol = similar(s_row, n_col)
    w = similar(X, n_row)
    zn = collect(1:n)    
    zncol = collect(1:n_col)
    zX = similar(X, n_row, n_col)
    zY = similar(X, n_row, q)
    pred = similar(X, n, 1)
    @inbounds for i in 1:B
        n_row == n ? srow .= zn : srow .= sample(1:n, n_row; replace = withr)
        n_col == p ? scol .= zncol : scol .= sample(1:p, n_col; replace = false)
        if i == 1
            r_array[:, :, 1] .= Y
        end
        zX .= @view(X[srow, scol])
        zY .= @view(r_array[srow, :, i])
        if isnothing(weights)
            fm[i] = flearn(zX, zY; kwargs...)
        else
            w .= mweights(weights[srow])
            fm[i] = flearn(zX, zY, w; kwargs...)
        end
        # @view is not accepted by XGBoost.predict
        # @view(X[:, scol])
        pred .= predict(fm[i], X[:, scol]).pred 
        r_array[:, :, i + 1] .= residreg(pred, r_array[:, :, i])
        s_row[:, i] .= srow
        s_col[:, i] .= scol
    end
    Gboostr(fm, s_row, s_col, r_array, nu)
end

function predict(object::Gboostr, X)
    B = size(object.r_array, 3) - 1
    scol = vcol(object.s_col, 1)
    nu = object.nu
    # @view is not accepted by XGBoost.predict
    # @view(X[:, scol])         
    acc = predict(object.fm[1], X[:, scol]).pred
    if B > 1
        @inbounds for i = 2:B
            scol = vcol(object.s_col, i)
            acc .+= nu * predict(object.fm[i], X[:, scol]).pred
        end
    end
    (pred = acc,)
end





