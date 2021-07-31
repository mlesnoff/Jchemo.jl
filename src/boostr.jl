struct Boostr
    fm
    s_obs
    s_var
    beta
    probs
end

""" 
    boostr(X, Y, weights = nothing; fun, B, 
        k = size(X, 1), withr = false, nvar = size(X, 2), meth = "dru", kwargs...)
Adaptative boosting (sampling) for regression models.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `fun` : Name (string) of the function computing the model to boost.
* `B` : Nb. of boosting iterations.
* `k` : Nb. observations (rows) sub-sampled in `X` at each iteration.
* `withr`: Boolean defining the type of sampling of the observations when `k` < n 
    (`withr = false` => sampling without replacement).
* `nvar` : Nb. variables (columns) sampled in `X` at each iteration.
* `meth` : Method of adaptative boosting ("dru" or "rt").
* `kwargs` : Named arguments to pass in 'fun`.

Assume that `X` is (n, p).

**At each iteration**
* If `k` = n : The model is calibrated after a sampling with 
    replacement of n observations.
* If `k` < n : The model is calibrated after a sampling of k observations
    without (default) or with replacement, depending o argument `withr`.
* If `nvar` < p , `nvar` variables are sampled without replacement, 
    and taken as predictors for the given iteration.
* Then, boosting weights are computed from the calibrated model 
    for the n observations and for the model.

This is the AdaBoost algorithm of Drucker 1997,
which is an adaptation of the AdaBoost.M1 classificatuon algorithm 
of Freund & Schapire 1997.

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
    k = size(X, 1), withr = false, nvar = size(X, 2), meth = "dru", kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)
    learn = eval(Meta.parse(fun))    
    fm = list(B)
    k = min(k, n)
    nvar = min(nvar, p)
    s_obs = fill(1, (k, B))
    s_var = similar(s_obs, nvar, B) 
    sobs = similar(s_obs, k)
    svar = similar(s_obs, nvar)
    probs = similar(X, n, B)
    zprobs = ones(n) / n
    r2 = similar(X, n)
    d = similar(X, n)
    beta = similar(X, B)
    znvar = collect(1:nvar) 
    zX = similar(X, k, nvar)
    zY = similar(Y, k, q)
    w = similar(X, k)
    @inbounds for i = 1:B
        k == n ? withr = true : nothing
        sobs .= sample(1:n, Weights(zprobs), k; replace = withr)
        nvar == p ? svar .= znvar : svar .= sample(1:p, nvar; replace = false)
        zX .= X[sobs, svar]
        zY .= Y[sobs, :]       
        if(isnothing(weights))
            fm[i] = learn(zX, zY; kwargs...)
        else
            w .= mweights(weights[sobs])
            fm[i] = learn(zX, zY, w; kwargs...)
        end
        pred = predict(fm[i], @view(X[:, svar])).pred
        r2 .= vec(sum(residreg(pred, Y).^2, dims = 2))
        d .= r2 / maximum(r2)[1]               
        L = dot(zprobs, d)
        beta[i] = L / (1 - L)
        if L <= .5                                  # or beta[i] >= 1
            zprobs .= mweights(beta[i].^(1 .- d))
        else
            zprobs .= ones(n) / n
        end
        s_obs[:, i] .= sobs
        s_var[:, i] .= svar
        probs[:, i] .= zprobs
    end
    Boostr(fm, s_obs, s_var, beta, probs)
end

function predict(object::Boostr, X)
    B = length(object.fm)
    svar = vcol(object.s_var, 1)
    w = log.(1 ./ object.beta)
    w .= mweights(w)
    acc = w[1] * predict(object.fm[1], @view(X[:, svar])).pred
    @inbounds for i = 2:B
        svar = vcol(object.s_var, i)
        acc .+= w[i] * predict(object.fm[i], @view(X[:, svar])).pred
    end
    (pred = acc,)
end

################ Direct weighting

""" 
    boostrw(X, Y, weights = nothing; fun, B, 
        k = size(X, 1), withr = false, nvar = size(X, 2), meth = "dru", kwargs...)
Adaptative boosting (direct) for regression models.
* `X` : X-data.
* `Y` : Y-data.
* `fun` : Name (string) of the function computing the model to boost.
    Must have a weight argument.
* `B` : Nb. of boosting iterations.
* `k` : Nb. observations (rows) sampled in `X` at each iteration.
* `withr`: Boolean defining the type of sampling of the observations when `k` < n 
    (`withr = false` => sampling without replacement).
* `nvar` : Nb. variables (columns) sub-sampled in `X` at each iteration.
* `meth` : Method of adaptative boosting ("dru" or "rt").
* `kwargs` : Named arguments to pass in 'fun`.

Same as `boostr` except that the boosting weights computed for the 
n observations are directly taken into account into the boosted model 
(there is no sampling of observations).
""" 
function boostrw(X, Y; fun, B, 
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
    s_obs = fill(1, (k, B))
    s_var = similar(s_obs, nvar, B) 
    sobs = similar(s_obs, k)
    svar = similar(s_obs, nvar)
    probs = similar(X, n, B)
    zprobs = ones(n) / n
    r2 = similar(X, n)
    d = similar(X, n)
    beta = similar(X, B)
    zX = similar(X, k, nvar)
    zY = similar(Y, k, q)
    zn = collect(1:n)
    znvar = collect(1:nvar) 
    @inbounds for i = 1:B
        k == n ? sobs .= zn : sobs .= sample(1:n, k; replace = withr)
        nvar == p ? svar .= znvar : svar .= sample(1:p, nvar; replace = false)
        zX .= X[sobs, svar]
        zY .= Y[sobs, :]       
        fm[i] = learn(zX, zY, zprobs[sobs]; kwargs...)
        pred = predict(fm[i], @view(X[:, svar])).pred
        r2 .= vec(sum(residreg(pred, Y).^2, dims = 2))
        d .= r2 / maximum(r2)[1]               
        L = dot(zprobs, d)
        beta[i] = L / (1 - L)
        if L <= .5                                    
            zprobs .= mweights(beta[i].^(1 .- d))
        else
            zprobs .= ones(n) / n
        end    
        s_obs[:, i] .= sobs
        s_var[:, i] .= svar
        probs[:, i] .= zprobs
    end
    Boostr(fm, s_obs, s_var, beta, probs)
end


