""" 
    getknn(Xtrain, X; metric = :eucl, k = 1)
Return the k nearest neighbors in `Xtrain` of each row 
    of the query `X`.
* `Xtrain` : Training X-data.
* `X` : Query X-data.
Keyword arguments:
* `metric` : Type of distance used for the query. 
    Possible values are ':eucl' (Euclidean),
        ':mah' (Mahalanobis).
* `k` : Number of neighbors to return.

The distances (not squared) are also returned.

## Examples
```julia
Xtrain = rand(5, 3)
X = rand(2, 3)
x = X[1:1, :]

k = 3
res = getknn(Xtrain, X; k)
res.ind  # indexes
res.d    # distances

res = getknn(Xtrain, x; k)
res.ind

res = getknn(Xtrain, X; metric = :mah, k)
res.ind
```
""" 
function getknn(Xtrain, X; metric = :eucl, k = 1)
    @assert in([:eucl, :mah])(metric) "Wrong value for argument 'metric'."
    Xtrain = ensure_mat(Xtrain)
    X = ensure_mat(X)
    n, p = size(Xtrain)
    k > n ? k = n : nothing
    if metric == :eucl
        ztree = NearestNeighbors.BruteTree(Matrix(Xtrain'), Euclidean())
        ind, d = NearestNeighbors.knn(ztree, Matrix(X'), k, true)    # 'ind' and 'd' are lists 
    elseif metric == :mah
        S = Statistics.cov(Xtrain, corrected = false)
        if p == 1
            Uinv = inv(sqrt(S)) 
        else
            if isposdef(S) == false
                Uinv = Diagonal(1 ./ diag(S))
            else
                Uinv = LinearAlgebra.inv!(cholesky!(Hermitian(S)).U)
            end
        end
        ## Below, since ztree = BruteTree(Xtrain', Mahalanobis(Sinv))
        ## is very slow:
        zXtrain = Xtrain * Uinv
        zX = X * Uinv
        ztree = NearestNeighbors.BruteTree(Matrix(zXtrain'), Euclidean())
        ind, d = NearestNeighbors.knn(ztree, Matrix(zX'), k, true)
    end
    #ind = reduce(hcat, ind)'
    (ind = ind, d)
end

