""" 
    getknn(Xtrain, X; k = 1, metric = "eucl")
Return the k nearest neighbors in Xtrain of each row of `X`.
* `Xtrain` : Training X-data.
* `X` : Query.
* `metric` : Type of distance used for the query. 
    Possible values are "eucl" or "mahal".

The distances are also returned.
""" 
function getknn(Xtrain, X; k = 1, metric = "eucl")
    Xtrain = ensure_mat(Xtrain)
    Xt = ensure_mat(X')
    n, p = size(Xtrain)
    k > n ? k = n : nothing
    if metric == "eucl"
        ztree = BruteTree(Xtrain', Euclidean())
        ind, d = knn(ztree, Xt, k, true) 
    elseif metric == "mahal"
        S = Statistics.cov(Xtrain, corrected = false)
        if p == 1
            U = sqrt(S) 
        else
            U = cholesky(Hermitian(S)).U
        end
        Uinv = inv(U)
        zXtrain = Xtrain * Uinv
        zX = X * Uinv
        ztree = BruteTree(zXtrain', Euclidean())
        # ztree = BruteTree(Xtraint, Mahalanobis(Sinv))
        # is very slow
        ind, d = knn(ztree, zX', k, true) 
    end
    #ind = reduce(hcat, ind)'
    (ind = ind, d = d)
end








