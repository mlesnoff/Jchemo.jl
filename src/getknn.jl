""" 
    getknn(Xtrain, X; metric = :eucl, k = 1)
Return the k nearest neighbors in `Xtrain` of each row of the query `X`.
* `Xtrain` : Training X-data.
* `X` : Query X-data.
Keyword arguments:
* `metric` : Type of distance used for the query. Possible values are `:eucl` (Euclidean), `:mah` (Mahalanobis), 
    `:sam` (a spectral angular distance), `:cos` (cosine distance), `:cor` (a correlation distance).
* `k` : Number of neighbors to return.

In addition to neighbors, the function also returns the distances (not squared). 

Consider two vectors x and y, and costheta = x'y / (||x|| ||y||) (theta is the angle between x and y, in radiants). 
In this function, the angular distances between x and y are computed as:
* Spectral angular distance (x, y) = acos(costheta) / pi    ==> [0, 1] 
* Cosine distance (x, y) = 1 - costheta                     ==> [0, 2]
* Correlation distance (x, y) = sqrt((1 - corr(x, y)) / 2)  ==> [0, 1]

Note: If x and y are centered, corr(x, y) = costheta.

## Examples
```julia
using Jchemo
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
    @assert in([:eucl, :mah, :cos, :sam, :cor])(metric) "Wrong value for argument 'metric'."
    Xtrain = ensure_mat(Xtrain)
    X = ensure_mat(X)
    n, p = size(Xtrain)
    k > n ? k = n : nothing
    if metric == :eucl
        tree = NearestNeighbors.BruteTree(Xtrain', Distances.Euclidean())
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
        ## Below: since tree = BruteTree(Xtrain', Mahalanobis(Sinv)) is very slow:
        Xtrain = Xtrain * Uinv
        X = X * Uinv
        tree = NearestNeighbors.BruteTree(Xtrain', Distances.Euclidean())
    elseif metric == :cos
        tree = NearestNeighbors.BruteTree(Xtrain', Distances.CosineDist())
    elseif metric == :sam
        tree = NearestNeighbors.BruteTree(Xtrain', Jchemo.SamDist())
    elseif metric == :cor
        tree = NearestNeighbors.BruteTree(Xtrain', Jchemo.CorDist())
    end
    ind, d = NearestNeighbors.knn(tree, X', k, true)     # 'ind' and 'd' are lists  
    ## Alternative: ind = reduce(hcat, ind)'
    (ind = ind, d)
end

