""" 
    baggr_vi(object::Baggr, X, Y; score = msep)
Compute variance importances (with permutation method) after bagging a regression model.
* `object` : Output of a bagging.
* `X` : X-data that was used in the model bagging.
* `Y` : Y-data that was used in the model bagging.

Variances importances are computed 
by permuting sucessively each column of the out-of-bag (X_OOB),
and by looking at the effect on the MSEP.   

## References

Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. 
https://doi.org/10.1023/A:1010933404324

Genuer, R., 2010. Forêts aléatoires : aspects théoriques, 
sélection de variables et applications. PhD Thesis. Université Paris Sud - Paris XI.

Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression. PhD Thesis. 
Univ. Paris 11. http://www.theses.fr/2002PA112245
""" 
function baggr_vi(object::Baggr, X, Y; score = msep)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    p = size(X, 2)
    q = size(Y, 2)
    B = length(object.s_oob)
    ncol = size(object.s_col, 1)
    scol = similar(object.s_col, ncol)
    res = similar(X, p, B)
    @inbounds for i = 1:B
        soob = object.s_oob[i]  # soob has a variable length
        m = length(soob)
        scol .= vcol(object.s_col, i)
        zpred = predict(object.fm[i], X[soob, scol]).pred
        zY = @view(Y[soob, :]) ;
        zscore = score(zpred, zY)
        zX = similar(X[soob, :])
        @inbounds for j = 1:p
            zX .= X[soob, :]
            s = sample(1:m, m, replace = false)
            zX[:, j] .= zX[s, j]
            zpred .= predict(object.fm[i], zX[:, scol]).pred
            zscore_perm = score(zpred, zY)
            if q == 1
                res[j, i] = (zscore_perm - zscore)[1]
            else
                res[j, i] = mean(zscore_perm .- zscore, dims = 2)[1]  
            end
        end
    end
    imp = vec(mean(res, dims = 2))
    (imp = imp, res = res)
end


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
function baggr_oob(object::Baggr, X, Y; score = msep)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = size(X, 1)
    q = size(Y, 2)
    B = length(object.fm)
    pred_oob = similar(X, n, q)
    k = zeros(n)
    @inbounds for i = 1:n
        zpred = fill(0., 1, q)
        zk = 0        
        @inbounds for j = 1:B
            soob = object.s_oob[j]
            if in(i, soob)
                zpred .+= predict(object.fm[j], X[i:i, vcol(object.s_col, j)]).pred
                zk += 1
            end
        end
        pred_oob[i, :] = zpred / zk
        k[i] = zk
    end
    scor = score(pred_oob, Y)
    (scor = scor, pred_oob, k)
end

