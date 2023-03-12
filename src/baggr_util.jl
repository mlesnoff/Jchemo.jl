""" 
    oob_baggr(object::Baggr, X, Y; score = rmsep)
Compute the out-of-bag (OOB) error after bagging a regression model.
* `object` : Output of a bagging.
* `X` : X-data used in the bagging.
* `Y` : Y-data used in the bagging.
* `score` : Function computing the prediction error (default: msep).
See `?baggr` for examples.
    
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
function oob_baggr(object::Baggr, X, Y; score = rmsep)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = nro(X)
    q = nco(Y)
    rep = length(object.fm)
    pred_oob = similar(X, n, q)
    k = zeros(n)
    @inbounds for i = 1:n
        zpred = zeros(1, q)
        zk = 0        
        @inbounds for j = 1:rep
            soob = object.soob[j]
            if in(i, soob)
                zpred .+= predict(object.fm[j], X[i:i, object.scol[j]]).pred
                zk += 1
            end
        end
        pred_oob[i, :] = zpred / zk
        k[i] = zk
    end
    s = findall(k .== 0)
    scor = score(rmrow(pred_oob, s), rmrow(Y, s))
    (scor = scor, pred_oob, k)
end

""" 
    vi_baggr(object::Baggr, X, Y; score = rmsep)
Variable importance (Out-of-bag permutation method).
* `object` : Output of a bagging.
* `X` : X-data that was used in the model bagging.
* `Y` : Y-data that was used in the model bagging.
Variances importances are computed 
by permuting sucessively each column of the out-of-bag (X_OOB),
and by looking at the effect on the error rate (e.g. RMSEP).  
See `?baggr` for examples.

## References
Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. 
https://doi.org/10.1023/A:1010933404324
Genuer, R., 2010. Forêts aléatoires : aspects théoriques, 
sélection de variables et applications. PhD Thesis. Université Paris Sud - Paris XI.
Gey, S., 2002. Bornes de risque, détection de ruptures, boosting : 
trois thèmes statistiques autour de CART en régression. PhD Thesis. 
Univ. Paris 11. http://www.theses.fr/2002PA112245
""" 
function vi_baggr(object::Baggr, X, Y; score = rmsep)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    p = nco(X)
    q = nco(Y)
    rep = length(object.soob)        # nb. bootstrap replications
    ncol = length(object.scol[1])    # nb. variables selected in each of the replications
    scol = similar(object.scol[1], ncol)
    res = similar(X, p, q, rep)
    @inbounds for i = 1:rep
        soob = object.soob[i]   # vector of variable length (OOB)
        scol .= object.scol[i]  # vector of consistent length
        m = length(soob)        # nb. OOB obs. for the given replication 
        ## OOB prediction
        ## fm[i] has been fitted on the observations not 
        ## in soob[i]and on columns in scol
        zpred = predict(object.fm[i], X[soob, scol]).pred
        zY = vrow(Y, soob)
        score0 = score(zpred, zY)   # reference (i.e. with no permutation) OOB score
        zX = similar(X, m, p)
        @inbounds for j = 1:p       
            zX .= X[soob, :]
            ## Permutation of the rows
            s = sample(1:m, m, replace = false)    #  
            zX[:, j] .= zX[s, j]
            ## End
            zpred .= predict(object.fm[i], zX[:, scol]).pred
            zscore = score(zpred, zY)
            if in(scol).(j)
                res[j, :, i] = zscore .- score0
            else
                res[j, :, i] .= NaN
            end
        end
    end
    ## Old: imp = reshape(mean(res, dims = 3), p, q)
    cnt = (!isnan).(res)
    res[isnan.(res)] .= 0
    cnttot = reshape(sum(cnt, dims = 3), p, q)
    restot = reshape(sum(res, dims = 3), p, q)
    imp = reshape(restot ./ cnttot, p, q)
    (imp = imp, cnt, res, cnttot, restot)
end


