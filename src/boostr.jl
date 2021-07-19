struct Boostr1
    fm
    s_obs
    s_var
end

""" 
    boostr(X, Y, weights = nothing ; fun, B, 
        k = size(X, 1), nvar = size(X, 2), kwargs...)
Bagging for regression models.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `fun` : Name (string) of the function computing the model to bagg.
* `B` : Nb. of sampling repetitions.
* `k` : Nb. observations (rows) sub-sampled in `X`.
* `nvar` : Nb. variables (columns) sub-sampled in `X`.
* `kwargs` : Named arguments to pass in 'fun`.

Assume that `X` is (n, p).

If `k` = n, each repetition consists in a sampling of the observations 
with replacement (this is the usual bagging).
If `k` < n, a sampling of k observations is done without replacement.

If `nvar` < p , `nvar` variables are sampled without replacement at each
boosting iteration, and taken as predictors for the given iteration.

## References

Breiman, L., 1996. Bagging predictors. Mach Learn 24, 123–140. 
https://doi.org/10.1007/BF00058655

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245
""" 
function boostr(X, Y, weights = nothing; fun, B, 
    k = size(X, 1), nvar = size(X, 2), kwargs...)
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
    zY = similar(Y, k, q)
    s_obs = fill(1, (k, B))
    s_var = similar(s_obs, nvar, B) 
    sobs = similar(s_obs, k)
    svar = similar(s_obs, nvar)
    w = similar(X, k)
    znvar = collect(1:nvar) 
    @inbounds for i = 1:B
        k == n ? repl = true : repl = false
        sobs .= sample(1:n, k; replace = repl)
        nvar == p ? svar .= znvar : svar .= sample(1:p, nvar; replace = false)
        zX .= X[sobs, svar]
        zY .= Y[sobs, :]
        if(isnothing(weights))
            fm[i] = learn(zX, zY; kwargs...)
        else
            w .= weights[sobs]
            w .= w / sum(w)
            fm[i] = learn(zX, zY, w; kwargs...)
        end
        s_obs[:, i] .= sobs
        s_var[:, i] .= svar
    end
    Boostr1(fm, s_obs, s_var)
end

function predict(object::Boostr1, X)
    B = length(object.fm)
    svar = vcol(object.s_var, 1)
    acc = predict(object.fm[1], @view(X[:, svar])).pred
    @inbounds for i = 2:B
        svar = vcol(object.s_var, i)
        acc .+= predict(object.fm[i], @view(X[:, svar])).pred
    end
    pred = acc ./ B
    (pred = pred,)
end




