
""" 
    baggr_oob(object::Baggr, X, Y; score = msep)
Compute the out-of-bag (OOB) error after bagging a regression model.
* `object` : Output of a bagging.
* `X` : X-data used in the bagging.
* `Y` : Y-data used in the bagging.
* `score` : Function computing the prediction error (default: msep).

## References

Breiman, L., 1996. Bagging predictors. Mach Learn 24, 123–140. 
https://doi.org/10.1007/BF00058655

Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. 
https://doi.org/10.1023/A:1010933404324

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, 
sélection de variables et applications. PhD Thesis. Université Paris Sud - Paris XI.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression (These de doctorat). 
Paris 11. http://www.theses.fr/2002PA112245
""" 
function baggr_oob(object::Baggr, X, Y; score)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    q = size(Y, 2)
    B = length(object.fm)
    nvar = size(object.s_var, 1)
    svar = fill(1, nvar)
    res = similar(X, B, q)
    @inbounds for i = 1:B
        soob = object.s_oob[i]
        svar .= vcol(object.s_var, i)
        zX = @view(X[soob, svar])
        zY = @view(Y[soob, :])
        zpred = predict(object.fm[i], zX).pred
        res[i, :] = score(zpred, zY)
    end
    mean(res, dims = 1)
end

""" 
    baggr_vi(object::Baggr, X, Y)
Compute variance importances (permutation method) after bagging a regression model.
* `object` : Output of a bagging.
* `X` : X-data used in the bagging.
* `Y` : Y-data used in the bagging.

Variances importances are computed 
by permuting sucessively each column of the out-of-bag (X_OOB)
and looking at the effect on the MSEP.   
## References

Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. 
https://doi.org/10.1023/A:1010933404324

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, 
sélection de variables et applications. PhD Thesis. Université Paris Sud - Paris XI.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression. PhD Thesis. 
Univ. Paris 11. http://www.theses.fr/2002PA112245
""" 
function baggr_vi(object::Baggr, X, Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    p = size(X, 2)
    s_oob = object.s_oob
    B = length(s_oob)
    nvar = size(object.s_var, 1)
    svar = similar(object.s_var, nvar)
    res = similar(X, B, p)
    @inbounds for i = 1:B
        soob = s_oob[i]
        m = length(soob)
        svar .= vcol(object.s_var, i)
        zpred = predict(object.fm[i], X[soob, svar]).pred
        zY = @view(Y[soob, :]) ;
        zscore = msep(zpred, zY)
        @inbounds for j = 1:p
            zX = copy(X[soob, :])
            s = sample(1:m, m, replace = false)
            zX[:, j] .= zX[s, j]
            zpred .= predict(object.fm[i], zX[:, svar]).pred
            zscore_perm = msep(zpred, zY)
            res[i, j] = mean(zscore_perm .- zscore, dims = 2)[1]
        end
    end
    imp = vec(mean(res, dims = 1))
    (imp = imp, res = res)
end








function baggr_vi2(X, Y, weights = nothing; fun, B, 
    k = size(X, 1), withr = false, nvar = size(X, 2), kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = size(X, 1)
    p = size(X, 2)
    #fm = baggr(X, Y, weights; fun = fun, B = B, k = k, withr = true,
    #    nvar = nvar, kwargs...)

    fm = baggr(X, Y, weights; fun = "treer", B = B, k = n, withr = true,
        nvar = p, kwargs...)

    s_oob = fm.s_oob
    svar = fill(1, nvar)
    res = similar(X, B, p)
    @inbounds for i = 1:B
        soob = s_oob[i]
        m = length(soob)
        svar .= vcol(fm.s_var, i)
        zpred = predict(fm.fm[i], X[soob, svar]).pred
        zY = Y[soob, :] ;
        zscore = msep(zpred, zY)
        #println(typeof(fm.fm[i]))
        #println(svar)
        @inbounds for j = 1:p
            #zX = @view(X[soob, svar])
            zX = copy(X[soob, svar])
            zX_perm = copy(X[soob, svar])            
            s = sample(1:m, m, replace = false)
            zX_perm[:, j] .= zX[s, j]

            zpred_perm = predict(fm.fm[i], zX_perm).pred
            zscore_perm = msep(zpred_perm, zY)
            res[i, j] = mean(zscore_perm .- zscore, dims = 2)[1]
        end
    end
    imp = vec(mean(res, dims = 1))
    (imp = imp, res = res)
end






