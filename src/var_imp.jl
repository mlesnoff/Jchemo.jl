function varimp_perm(Xtrain, Ytrain, X, Y; score = msep, fun, B, kwargs...)
    X = ensure_mat(X)
    m, p = size(X)
    fm = fun(Xtrain, Ytrain; kwargs...)
    pred = predict(fm, X).pred
    zscore = score(pred, Y)
    zX = similar(X)     
    res = similar(X, B, p)
    @inbounds for j = 1:p
        zX .= X
        @inbounds for i = 1:B
            s = StatsBase.sample(1:m, m, replace = false)
            zX[:, j] .= zX[s, j]
            pred .= predict(fm, zX).pred
            zscore_perm = score(pred, Y)
            res[i, j] = mean(zscore_perm .- zscore, dims = 2)[1]
        end
    end
    imp = vec(mean(res, dims = 1))
    (imp = imp, res = res)
end 

function varimp_chisq(X, Y; probs = [.25; .75])
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    p = size(X, 2)
    q = size(Y, 2)
    zX = similar(X)
    zY = similar(Y)
    imp = similar(X, p, q)
    @inbounds for j = 1:p
        z = vcol(X, j)
        quants = Statistics.quantile(z, probs)
        zX[:, j] .= recod_cont2cla(z, quants)
    end
    @inbounds for j = 1:q
        z = vcol(Y, j)
        quants = Statistics.quantile(z, probs)
        zY[:, j] .= recod_cont2cla(z, quants)
    end
    zX = Int64.(zX)
    zY = Int64.(zY)
    @inbounds for i = 1:q
        zy = vcol(zY, i)
        @inbounds for j = 1:p
            z = StatsBase.counts(vcol(zX, j), zy)
            res = HypothesisTests.ChisqTest(z)
            imp[j, i] = res.stat
            #imp[j, i] = 1 - exp(-res.stat)
            #imp[j, i] = 1 - pvalue(res)
        end
    end
    (imp = imp,)
end 









