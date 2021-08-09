
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
    ncol = size(object.s_col, 1)
    scol = fill(1, ncol)
    res = similar(X, B, q)
    @inbounds for i = 1:B
        soob = object.s_oob[i]
        scol .= vcol(object.s_col, i)
        # @view is not accepted by XGBoost.predict
        # @view(X[soob, scol])
        # @view(Y[soob, :])
        zX = X[soob, scol]
        zY = Y[soob, :]
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
    ncol = size(object.s_col, 1)
    scol = similar(object.s_col, ncol)
    res = similar(X, B, p)
    @inbounds for i = 1:B
        soob = s_oob[i]
        m = length(soob)
        scol .= vcol(object.s_col, i)
        zpred = predict(object.fm[i], X[soob, scol]).pred
        zY = @view(Y[soob, :]) ;
        zscore = msep(zpred, zY)
        @inbounds for j = 1:p
            zX = copy(X[soob, :])
            s = sample(1:m, m, replace = false)
            zX[:, j] .= zX[s, j]
            zpred .= predict(object.fm[i], zX[:, scol]).pred
            zscore_perm = msep(zpred, zY)
            res[i, j] = mean(zscore_perm .- zscore, dims = 2)[1]
        end
    end
    imp = vec(mean(res, dims = 1))
    (imp = imp, res = res)
end







