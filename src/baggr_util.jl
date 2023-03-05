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
    ## Nb. bootstrap replications
    rep = length(object.soob)
    ## Variables selected in each of the replications
    nscol = length(object.scol[1])
    scol = similar(object.scol[1], nscol)
    ## End
    res = similar(X, p, q, rep)
    @inbounds for i = 1:rep
        soob = object.soob[i]   # vector of variable length (OOB)
        scol .= object.scol[i]  # vector of consistent length
        m = length(soob)
        ## fm[i] has been fitted with observations out of OOB
        ## and only the columns in scol
        zpred = predict(object.fm[i], X[soob, scol]).pred
        zY = vrow(Y, soob)
        score0 = score(zpred, zY)   # reference score (i.e. with no permutation) for OOB
        zX = similar(X, m, p)
        ## Warning: The following 'for' should be limited to variables 
        ## in scol.The present version of the function is only valid 
        ## when scol = all the variables (i.e. 'colsamp = 1' in 'baggr')
        @inbounds for j = 1:p       
            zX .= X[soob, :]
            ## Permutation of the rows of OOB[i]
            s = sample(1:m, m, replace = false)    #  
            zX[:, j] .= zX[s, j]
            ## End
            zpred .= predict(object.fm[i], zX[:, scol]).pred
            zscore = score(zpred, zY)
            res[j, :, i] = zscore .- score0
        end
    end
    imp = reshape(mean(res, dims = 3), p, q)
    (imp = imp, res)
end


