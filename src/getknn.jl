"""
    getknn(Xtrain, X ; metric = "euclid", k = 1)
Returns the k nearest neighbors of each row of X (= observations)
in Xtrain. The corresponding distances are also returned.
""" 
function getknn(Xtrain, X ; metric = "euclid", k = 1)
    Xtrain = ensure_mat(Xtrain)
    X = ensure_mat(X)
    tree = NearestNeighbors.BruteTree
    if metric == "euclid"
        tXtrain = Xtrain'
        tX = X'
        ztree = tree(tXtrain, Euclidean())
    end
    if metric == "mahal"
    end
    ind, d = knn(ztree, tX, k, true)  
    #ind = reduce(hcat, ind)'
    (ind = ind, d = d)
end


