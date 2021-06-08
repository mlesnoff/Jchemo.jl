"""
    getknn(Xtrain, X ; k = 1, metric = "eucl")
Return the k nearest neighbors of each row of X (= observations) in Xtrain. 
The distances are also returned.
- metric: "eucl", "mahal"
""" 
function getknn(Xtrain, X ; k = 1, metric = "eucl")
    Xtrain = ensure_mat(Xtrain)
    Xt = ensure_mat(X')
    p = size(Xtrain, 2)
    if metric == "eucl"
        ztree = BruteTree(Xtrain', Euclidean())
        ind, d = knn(ztree, Xt, k, true) 
    end
    if metric == "mahal"
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
        ## ztree = BruteTree(Xtraint, Mahalanobis(Sinv))
        ## is very slow
        ind, d = knn(ztree, zX', k, true) 
    end
    #ind = reduce(hcat, ind)'
    (ind = ind, d = d)
end








