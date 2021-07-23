struct Gboostr
    fm
    s_obs
    s_var
    r_array
    nu
end

""" 
    gboostr(X, Y, weights = nothing; fun, B = 1, nu = 1, 
        k = size(X, 1), withr = false, nvar = size(X, 2), kwargs...)
Gradient boosting for regression models.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `fun` : Name (string) of the function computing the model to boost.
* `B` : Nb. of boosting iterations.
* `nu` : Learning rate (must be between 0 and 1) applied to each iteration.
* `k` : Nb. observations (rows) sub-sampled in `X` at each iteration.
* `withr`: Boolean defining the type of sampling of the observations when `k` < n 
    (`withr = false` => sampling without replacement).
* `nvar` : Nb. variables (columns) sub-sampled in `X` at each iteration.
* `kwargs` : Named arguments to pass in 'fun`.

Assume that `X` is (n, p).

If `k` = n, each boosting iteration takes all the observations (no sub-sampling),
which is the usual gradient boosting.

If `k` < n, each boosting iteration is done on k sub-sampled observations, 
which corresponds to the stochastic gradient boosting. 
The sampling can be without (default) or with replacement, depending on argument `withr`.

If `nvar` < p , `nvar` variables are sampled without replacement at each
boosting iteration, and taken as predictors for the given iteration.

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
    k = size(X, 1), withr = false, nvar = size(X, 2), kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)
    learn = eval(Meta.parse(fun))    
    fm = list(B)
    k = min(k, n)
    nvar = min(nvar, p)
    zX = similar(X, k, nvar)
    zY = similar(X, k, q)
    r_array = similar(X, n, q, B + 1) # residuals    
    s_obs = fill(1, (k, B))
    s_var = similar(s_obs, nvar, B)
    sobs = similar(s_obs, k)
    svar = similar(s_obs, nvar)
    w = similar(X, k)
    zn = collect(1:n)    
    znvar = collect(1:nvar)
    @inbounds for i in 1:B
        k == n ? sobs .= zn : sobs .= sample(1:n, k; replace = withr)
        nvar == p ? svar .= znvar : svar .= sample(1:p, nvar; replace = false)
        if i == 1
            r_array[:, :, 1] .= Y
        end
        zX .= @view(X[sobs, svar])
        zY .= @view(r_array[sobs, :, i])
        if(isnothing(weights))
            fm[i] = learn(zX, zY; kwargs...)
        else
            w .= mweights(weights[sobs])
            fm[i] = learn(zX, zY, w; kwargs...)
        end
        pred = predict(fm[i], @view(X[:, svar])).pred
        r_array[:, :, i + 1] .= residreg(pred, r_array[:, :, i])
        s_obs[:, i] .= sobs
        s_var[:, i] .= svar
    end
    Gboostr(fm, s_obs, s_var, r_array, nu)
end

function predict(object::Gboostr, X)
    B = size(object.r_array, 3) - 1
    svar = vcol(object.s_var, 1)
    nu = object.nu
    acc = predict(object.fm[1], @view(X[:, svar])).pred
    if B > 1
        @inbounds for i = 2:B
            svar = vcol(object.s_var, i)
            acc .+= nu * predict(object.fm[i], @view(X[:, svar])).pred
        end
    end
    (pred = acc,)
end





