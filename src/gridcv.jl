function gridcv(X, Y; segm, score, fun, pars, verbose = true)
    q = size(Y, 2)
    nrep = length(segm)
    res_rep = list(nrep)
    nco = length(pars[1]) # nb. combinations in pars
    @inbounds for i in 1:nrep
        verbose ? println("/ rep=", i, " ") : nothing
        listsegm = segm[i]       # segments in the repetition
        nsegm = length(listsegm) # segmts: 1, segmkf: K
        zres = list(nsegm)       # results for the repetition
        @inbounds for j = 1:nsegm
            verbose ? println("segm=", j, " ") : nothing
            s = sort(listsegm[j])
            zres[j] = gridscore(
                rmrows(X, s), rmrows(Y, s),
                X[s, :], Y[s, :];
                score = score, fun = fun, pars = pars)
        end
        zres = reduce(vcat, zres)
        dat = DataFrame(rep = fill(i, nsegm * nco),
            segm = repeat(1:nsegm, inner = nco))
        zres = hcat(dat, zres)
        res_rep[i] = zres
    end
    verbose ? println("/ End.") : nothing
    res_rep = reduce(vcat, res_rep)

    gdf = groupby(res_rep, collect(keys(pars))) 
    namy = map(string, repeat(["y"], q), 1:q)
    res = combine(gdf, namy .=> mean, renamecols = false)

    (res = res, res_rep = res_rep, )
end
    
















function gridcv_1comb(X, Y ; segm, score, fun, kwargs...)
 
    nrep = size(segm, 1)
    zscore = list(nrep)
    for i = 1:nrep 
        zsegm = segm[i]
        nsegm = size(zsegm, 1)
        pred = list(nsegm)
        Ytest = list(nsegm)
        for j = 1:nsegm
            s = zsegm[j]
            Xtrain = rmrow(X, s)
            Ytrain = rmrow(Y, s)
            Xtest = X[s, :]
            Ytest[j] = Y[s, :]
            fm = fun(Xtrain, Ytrain ; kwargs...)          
            pred[j] = predict(fm, Xtest)
        end
        pred = reduce(vcat, pred)
        Ytest = reduce(vcat, Ytest)
        zscore[i] = score(Ytest, pred)
     end   
     reduce(vcat, zscore)
     
end

function gridcv2(X, Y ;  doprint = true, segm, score, fun, kwargs...)

    res = map(values(kwargs)...) do v...
        if doprint == true
            println(Pair.(keys(kwargs), v))
        end
        gridcv_1comb(X, Y ; segm, score, fun, Pair.(keys(kwargs), v)...)
    end
    
    ncomb = size(res, 1)
    if ncomb == 1
        z = colmeans(res)
    else
        z = list(ncomb)
        for i = 1:ncomb
            z[i] = colmeans(res[i])
        end
        z = reduce(vcat, z)
    end
    
    (summ = z, rep = res)

end




