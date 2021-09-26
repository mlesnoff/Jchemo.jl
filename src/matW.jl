"""
    matB(X, group; fun = mean)
Compute the between covariance matrix ("B") of `X`.
* `X` : X-data (n, p).
* `group` : A vector (n,) defing the groups.
""" 
matB = function(X, group)
    X = ensure_mat(X)
    z = aggstat(X, group; fun = mean)
    B = matcov(z.res, mweights(z.ni))
    (B = B, ct = z.res, lev = z.lev, ni = z.ni)
end


"""
    matW(X, group; fun = mean)
Compute the within covariance matrix ("W") of `X`.
* `X` : X-data (n, p).
* `group` : A vector (n,) defing the groups.
""" 
matW = function(X, group)
    X = ensure_mat(X)
    ztab = tab(group)
    lev = ztab.keys
    nlev = length(lev)
    ni = collect(values(ztab))
    # Case with group(s) with only 1 obs
    sum(ni .== 1) > 0 ? sigma_1obs = matcov(X) : nothing
    # End
    w = mweights(ni)
    Wi = list(nlev)
    W = zeros(1, 1)
    for i in 1:nlev 
        if ni[i] == 1
            Wi[i] = sigma_1obs
        else
            s = findall(group .== lev[i])
            Wi[i] = matcov(X[s, :])
        end
        if(i == 1) 
            W = w[i] * Wi[i] 
        else 
            W = W + w[i] * Wi[i]
            # Alternative: Could give weight=0 to the class(es) with 1 obs
        end
    end
    (W = W, Wi = Wi, lev = lev, ni = ni)
end






