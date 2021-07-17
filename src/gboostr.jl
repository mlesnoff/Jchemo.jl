struct Gboostr
    fm
    s_obs
    s_var
    r_array
    nu
end

""" 
    gboostr(X, Y, weights = nothing; fun, nboost = 1, nu = 1, kwargs...)
Gradient boosting for regression models.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `fun` : Name (string) of the function computing the model to boost.
* `nboost` : Nb. of boosting.
* `nu` : Learning rate (must be between 0 and 1).
* k : Nb. observation sub-sampled.
* mtry : Nb. variables (columns) sub-sampled in `X`.
* `kwargs` : Named arguments to pass in 'fun`.

## References

Friedman, J.H., 2001. Greedy Function Approximation: A Gradient Boosting Machine. 
The Annals of Statistics 29, 1189–1232.

Friedman, J.H., 2002. Stochastic gradient boosting. 
Computational Statistics & Data Analysis, Nonlinear Methods and Data Mining 38, 367–378.
https://doi.org/10.1016/S0167-9473(01)00065-2

""" 
function gboostr(X, Y, weights = nothing; fun, nboost = 1, nu = 1, 
    k = size(X, 1), mtry = size(X, 2), kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)
    learn = eval(Meta.parse(fun))    
    fm = list(nboost)
    k = min(k, n)
    mtry = min(mtry, p)
    s_obs = fill(1, (k, nboost))
    s_var = similar(s_obs, mtry, nboost)
    r_array = similar(X, n, q, nboost + 1) # residuals    
    zX = similar(X, k, mtry)
    zY = similar(X, k, q)
    @inbounds for i in 1:nboost
        sobs = sample(1:n, k; replace = false)
        svar = sample(1:p, mtry; replace = false)
        if i == 1
            r_array[:, :, 1] .= Y
        end
        zX .= @view(X[sobs, svar])
        zY .= @view(r_array[sobs, :, i])
        if(isnothing(weights))
            fm[i] = learn(zX, zY; kwargs...)
        else
            w = weights[sobs]
            zweights = w / sum(w)
            fm[i] = learn(zX, zY, zweights; kwargs...)
        end
        pred = predict(fm[i], @view(X[:, svar])).pred
        r_array[:, :, i + 1] .= residreg(pred, r_array[:, :, i])
        s_obs[:, i] .= sobs
        s_var[:, i] .= svar
    end
    Gboostr(fm, s_obs, s_var, r_array, nu)
end

function predict(object::Gboostr, X)
    nboost = size(object.r_array, 3) - 1
    svar = vcol(object.s_var, 1)
    nu = object.nu
    acc = predict(object.fm[1], @view(X[:, svar])).pred
    if nboost > 1
        @inbounds for i = 2:nboost
            svar = vcol(object.s_var, i)
            acc .+= nu * predict(object.fm[i], @view(X[:, svar])).pred
        end
    end
    (pred = acc,)
end





