struct Boostr
    fm
    s_obs
    s_var
    beta
    probs
end

""" 
    boostr(X, Y, weights = nothing; fun, nboost, 
        k = size(X, 1), withr = false, nvar = size(X, 2), meth = "dru", kwargs...)
Adaptative boosting (by sampling) for regression models.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `fun` : Name (string) of the function computing the model to boost.
* `nboost` : Nb. of boosting iterations.
* `k` : Nb. observations (rows) sub-sampled in `X` at each iteration.
* `withr`: Boolean defining the type of sampling of the observations when `k` < n 
    (`withr = false` => sampling without replacement).
* `nvar` : Nb. variables (columns) sub-sampled in `X` at each iteration.
* `meth` : Method of adaptative boosting ("dru" or "rt").
* `kwargs` : Named arguments to pass in 'fun`.

Assume that `X` is (n, p).

If `k` = n, a sampling of n observations is done with replacement.

If `k` < n, a sampling of k observations is done without (default) 
or with replacement, depending o argument `withr`.

If `nvar` < p , `nvar` variables are sampled without replacement at each
iteration, and taken as predictors for the given iteration.

Methods
* `meth` = "dru" : This is the AdaBoost algorithm of Drucker 1997,
    which is an adaptation of the AdaBoost.M1 classificatuon algorithm 
    of Freund & Schapire 1997.
* `meth` = "rt" : This is the AdaBoost.RT algorithm of Shrestha & Solomatine 2006,
slightly modified (parameter phi is computed automatically frop a quantile).

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
function boostr(X, Y, weights = nothing; fun, nboost, 
    k = size(X, 1), withr = false, nvar = size(X, 2), meth = "dru", kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)
    learn = eval(Meta.parse(fun))    
    fm = list(nboost)
    k = min(k, n)
    nvar = min(nvar, p)
    s_obs = fill(1, (k, nboost))
    s_var = similar(s_obs, nvar, nboost) 
    sobs = similar(s_obs, k)
    svar = similar(s_obs, nvar)
    w = similar(X, k)
    probs = similar(X, n, nboost)
    zprobs = ones(n) / n
    r2 = similar(X, n)
    d = similar(X, n)
    beta = similar(X, nboost)
    znvar = collect(1:nvar) 
    zX = similar(X, k, nvar)
    zY = similar(Y, k, q)
    @inbounds for i = 1:nboost
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
        if isequal(meth, "dru")
            r2 .= vec(sum(residreg(pred, Y).^2, dims = 2))
            d .= r2 / maximum(r2)[1]               
            L = dot(zprobs, d)
            beta[i] = L / (1 - L)
            if L <= .5  # or beta[i] >= 1
                zprobs .= mweights(beta[i].^(1 .- d))
            else
                zprobs .= ones(n) / n
            end
        elseif isequal(meth, "rt")
            r2 .= vec(mean(abs.(residreg(pred, Y)) ./ Y, dims = 2))
            phi = quantile(r2, .25)        # r2 > phi ==> "non correct" prediction
            eps = sum(zprobs[r2 .> phi])   # binary error rate (prop. non correct pred.)
            alpha = 2
            beta[i] = eps^alpha
            zprobs[r2 .<= phi] .= beta[i]
            zprobs[r2 .> phi] .= 1
            zprobs .= mweights(zprobs)        
        end
        s_obs[:, i] .= sobs
        s_var[:, i] .= svar
        probs[:, i] .= zprobs
    end
    Boostr(fm, s_obs, s_var, beta, probs)
end

function predict(object::Boostr, X)
    nboost = length(object.fm)
    svar = vcol(object.s_var, 1)
    w = log.(1 ./ object.beta)
    w .= mweights(w)
    acc = w[1] * predict(object.fm[1], @view(X[:, svar])).pred
    @inbounds for i = 2:nboost
        svar = vcol(object.s_var, i)
        acc .+= w[i] * predict(object.fm[i], @view(X[:, svar])).pred
    end
    (pred = acc,)
end




