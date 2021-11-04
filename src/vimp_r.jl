function vimp_perm_r(Xtrain, Ytrain, X, Y; score = msep, fun, 
        perm = 50, kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    m, p = size(X)
    q = size(Y, 2)
    fm = fun(Xtrain, Ytrain; kwargs...)
    pred = predict(fm, X).pred
    zscore = score(pred, Y)
    zX = similar(X)     
    res = similar(X, p, perm, q)
    @inbounds for j = 1:p
        zX .= X
        @inbounds for i = 1:perm
            s = StatsBase.sample(1:m, m, replace = false)
            zX[:, j] .= zX[s, j]
            pred .= predict(fm, zX).pred
            zscore_perm = score(pred, Y)
            res[j, i, :] = zscore_perm .- zscore
        end
    end
    imp = reshape(mean(res, dims = 2), p, q)
    (imp = imp,)
end 

function vimp_chisq_r(X, Y; probs = [.25 ; .75])
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
        zX[:, j] .= recodnum2cla(z, quants)
    end
    @inbounds for j = 1:q
        z = vcol(Y, j)
        quants = Statistics.quantile(z, probs)
        zY[:, j] .= recodnum2cla(z, quants)
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

function vimp_aov_r(X, Y; probs = [.25 ; .75])
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    q = size(Y, 2)
    zy = similar(Y, n)
    imp = similar(X, p, q)
    @inbounds for j = 1:q
        z = vcol(Y, j)
        quants = Statistics.quantile(z, probs)        
        zy .= recodnum2cla(z, quants)
        imp[:, j] .= vec(aov1(zy, X).F)
    end
    (imp = imp,)
end 

function vimp_aov_r2(X, Y; probs = [.25 ; .75])
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    q = size(Y, 2)
    imp = similar(X, p, q)
    zX = similar(X)
    @inbounds for j = 1:p
        z = vcol(X, j)
        quants = Statistics.quantile(z, probs)        
        zX[:, j] .= recodnum2cla(z, quants)
        imp[j, :] .= vec(aov1(vcol(zX, j), Y).F)
    end
    (imp = imp,)
end 

function vimp_cov_r(X, Y)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    q = size(Y, 2)
    imp = similar(X, p, q)
    @inbounds for j = 1:q
        imp[:, j] .= vec(cov(vcol(Y, q), X).^2)
    end
    (imp = imp,)
end 



