"""
    center(X, v = colmeans(X)) 
Column-wise centering of X by v
X :
- Matrix (n, p), (n, 1), (1, p)
- Vector (n,)
v: Vector (n,)
""" 
function center(X, v = colmeans(X))    
    X .- v'
end

function center!(X, v = colmeans(X))
    p = size(X, 2)
    @inbounds for j = 1:p
        X[:, j] .= @view(X[:, j]) .- v[j]
        ## Same as
        ## @simd for i in 1:size(X, 1)
        ##     X[i, j] = X[i,j] - v[j]
        ## end
    end
    X
end

"""
    colmeans(X, weights)
Weighted mean of each column of X
X: 
- Matrix (n, p) ==> vector (p,)
- Vector(n,)  ==> scalar
w: Vector (n,)
""" 
function colmeans(X, weights = ones(size(X, 1)))
    xmeans = mweights(weights)' * X
    #length(size(xmeans)) > 1 ? xmeans = vec(xmeans) : nothing
    if length(size(xmeans)) > 1
        xmeans = vec(xmeans)
    end
    xmeans
end

"""
    ensure_mat(X::AbstractMatrix)
    ensure_mat(X::AbstractVector)
Reshape X to a matrix if necessary
"""
ensure_mat(X::AbstractMatrix) = X
ensure_mat(X::AbstractVector) = reshape(X, :, 1)
ensure_mat(X::Number) = reshape([X], 1, 1)

"""
    list(n::Integer)
Create a "list" of length n, i.e. a Vector{Any}(undef, n) 
"""  
list(n::Integer) = Vector{Any}(undef, n) 

""" 
    mweights(w)
Returns a vector of weights that sums to 1
w: a vector
"""
mweights(w) = w / sum(w)

"""
    row(X, j)
    col(X, j)
View on the i-th row or j-th column of a matrix
""" 
row(X, i) = view(X, :, i)
col(X, j) = view(X, :, j)



