""" 
    getknn(Xtrain, X; k = 1, metric = "eucl")
Return the k nearest neighbors in Xtrain of each row of `X`.
* `Xtrain` : Training X-data.
* `X` : Query X-dta.
* `metric` : Type of distance used for the query. 
    Possible values are "eucl" or "mahal".

The distances (not squared) are also returned.

## Examples
```julia
Xtrain = rand(5, 3)
X = rand(2, 3)
x = X[1:1, :]

k = 3
res = getknn(Xtrain, X; k = k)
res.ind  # indexes
res.d    # distances

res = getknn(Xtrain, x; k = k)
res.ind

res = getknn(Xtrain, X; k = k, metric = "mahal")
res.ind
```
""" 
function getknn(Xtrain, X; k = 1, metric = "eucl")
    Xtrain = ensure_mat(Xtrain)
    X = ensure_mat(X)
    n, p = size(Xtrain)
    k > n ? k = n : nothing
    if metric == "eucl"
        ztree = BruteTree(Matrix(Xtrain'), Euclidean())
        ind, d = knn(ztree, Matrix(X'), k, true) 
    elseif metric == "mahal"
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
        zXtrain = Xtrain * Uinv
        zX = X * Uinv
        ztree = BruteTree(Matrix(zXtrain'), Euclidean())
        # Since ztree = BruteTree(Xtraint, Mahalanobis(Sinv))
        # is very slow
        ind, d = knn(ztree, Matrix(zX'), k, true)    # ind and d = lists
    end
    #ind = reduce(hcat, ind)'
    (ind = ind, d)
end


