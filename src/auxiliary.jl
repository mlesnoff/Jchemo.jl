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
rmrows(X, collect(1:18))
rmrows(X, 1:18)
"""
function rmrows(X, s)
    n = size(X, 1)
    invs = setdiff(collect(1:n), s)
    X[invs, :]
end

function rmcols(X, s)
    p = size(X, 2)
    invs = setdiff(collect(1:p), s)
    X[:, invs]
end

"""
    row(X, j)
    col(X, j)
View on the i-th row or j-th column of a matrix
""" 
row(X, i) = view(X, :, i)
col(X, j) = view(X, :, j)



