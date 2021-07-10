"""
    ensure_mat(X)
Reshape `X` to a matrix if necessary.
"""
ensure_mat(X::AbstractMatrix) = X
ensure_mat(X::AbstractVector) = reshape(X, :, 1)
ensure_mat(X::Number) = reshape([X], 1, 1)

"""
    iqr(x)
IQR.
"""
iqr(x) = quantile(x, .75) - quantile(x, .25)


"""
    list(n::Integer)
Create a Vector{Any}(undef, n).
"""  
list(n::Integer) = Vector{Any}(undef, n) 

""" 
    mad(x)
Compute the MAD.
"""
mad(x) = 1.4826 * median(abs.(x .- median(x)))

""" 
    mweights(w)
Return a vector of weights that sums to 1.
* `w` : A vector (n,).
"""
mweights(w) = w / sum(w)

"""
    rmrow(X, s)
Remove the rows of `X` having indexes `s`.
## === Examples
```
X = rand(20, 4) ; 
rmrows(X, collect(1:18))
rmrows(X, 1:18)
```

"""
function rmrows(X::AbstractMatrix, s)
    n = size(X, 1)
    minus_s = setdiff(collect(1:n), s)
    X[minus_s, :]
end
function rmrows(X::AbstractVector, s)
    n = length(X)
    minus_s = setdiff(collect(1:n), s)
    X[minus_s]
end

"""
    rmcols(X, s)
Remove the columns of `X` having indexes `s`.
"""
function rmcols(X, s)
    p = size(X, 2)
    minus_s = setdiff(collect(1:p), s)
    X[:, minus_s]
end

"""
    vrow(X, j)
    vcol(X, j)
View of the i-th row or j-th column of a matrix `X`.
""" 
vrow(X, i) = view(X, i, :)
vcol(X, j) = view(X, :, j)



