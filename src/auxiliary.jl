"""
    center(X, v = colmeans(X)) 
Column-wise centering of X by v
X : matrix (n, p) or vector (n,)
v: vector (n,)
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
- matrix (n, p) ==> vector (p,)
- vector(n,)  ==> scalar
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
Reshape X to a matrix if necessary.
"""
ensure_mat(X::AbstractMatrix) = X
ensure_mat(X::AbstractVector) = reshape(X, :, 1)
ensure_mat(X::Number) = reshape([X], 1, 1)

"""
    list(n::Integer)
Create a Vector{Any}(undef, n).
"""  
list(n::Integer) = Vector{Any}(undef, n) 

""" 
    mad(x)
"""
mad(x) = 1.4826 * median(abs.(x .- median(x)))

""" 
    mweights(w)
Return a vector of weights that sums to 1.
w: a vector
"""
mweights(w) = w / sum(w)

"""
   rmrow(X, s)
Remove the rows of X having indexes s.
Examples
≡≡≡≡≡≡≡≡≡≡
X = rand(20, 4) ; 
rmrow(X, collect(1:18))
rmrow(X, 1:18)
"""
function rmrow(X, s)
    n = size(X, 1)
    invs = setdiff(collect(1:n), s)
    X[invs, :]
end

"""
    row(X, j)
    col(X, j)
View on the i-th row or j-th column of a matrix
""" 
row(X, i) = view(X, :, i)
col(X, j) = view(X, :, j)



