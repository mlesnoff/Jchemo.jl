"""
    colmeans(X)
    colmeans(X, w)
Weighted mean of each column of X
- X: matrix (n, p) or vector (p,)
- w: Vector (n,)
Return a vector.
Note: For true mean, w must be preliminary "normalized" to sum to 1.
""" 
colmeans(X) = vec(Statistics.mean(X; dims = 1))

colmeans(X, w) = vec(w' * ensure_mat(X))

"""
    colvars(X)
    colvars(X, w)
Weighted variance of each column of X
- X: matrix (n, p) or vector (p,)
- w: Vector (n,)
Return a vector, or scalar.
Note: For true variance, w must be preliminary "normalized" to sum to 1.
""" 
colvars(X) = vec(Statistics.var(X; corrected = false, dims = 1))

colvars(X, w) = colvars!(copy(X), w)
function colvars!(X, w)
    xmeans = colmeans(X, w)             # Consumes allocation. Could be colmeans2(X, w) but slower
    center!(X, xmeans)
    vec(w' * (ensure_mat(X).^2))        # Try to replace X.^2 with boucle and mutation of X[i, j]
end

"""
    center(X)
    center!(X)
    center(X, v) 
    center!(X, v)
Column-wise centering of X by v
- X: matrix (n, p) or vector (n,)
- v: vector (p,)
""" 
center(X) = center!(copy(X))
center!(X) = center!(X, colmeans(X))

center(X, v) = center!(copy(X), v)
function center!(X, v)
    p = size(X, 2)
    @inbounds for j = 1:p
        X[:, j] .= col(X, j) .- v[j]
    end
    X 
end

"""
    scale(X)
    scale!(X)
    scale(X, v)
    scale!(X, v) 
Column-wise scaling of X by v
- X: matrix (n, p) or vector (n,)
- v: vector (p,)
""" 
scale(X) = scale!(copy(X))
scale!(X) = scale!(X, sqrt.(colvars(X)))

scale(X, v) = scale!(copy(X), v)
function scale!(X, v)
    p = size(X, 2)
    @inbounds for j = 1:p
        X[:, j] .= col(X, j) ./ v[j]
    end
    X 
end



