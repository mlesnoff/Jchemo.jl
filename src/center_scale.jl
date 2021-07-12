"""
    colmeans(X)
    colmeans(X, w)
Compute the mean of each column of `X`.
* `X` : Data.
* `w` : Weights of the observations.

Return a vector.

For a true mean, `w` must preliminary be normalized to sum to 1.
""" 
colmeans(X) = vec(Statistics.mean(X; dims = 1))

colmeans(X, w) = vec(w' * ensure_mat(X))

"""
    colvars(X)
    colvars(X, w)
Compute the (uncorrected) variance of each column of `X`.
* `X` : Data.
* `w` : Weights of the observations.

Return a vector.

**Note:** For a true variance, `w` must preliminary be normalized to sum to 1.
""" 
colvars(X) = vec(Statistics.var(X; corrected = false, dims = 1))

function colvars(X, w)
    p = size(X, 2)
    z = colmeans(X, w)
    @inbounds for j = 1:p
        z[j] = dot(view(w, :), (vcol(X, j) .- z[j]).^2)        
    end
    z 
end

"""
    center(X, v) 
Center each column of `X`.
* `X` : Data.
* `v` : Centering factors.
""" 
function center(X, v)
    M = copy(X)
    center!(M, v)
    M
end

function center!(X, v)
    p = size(X, 2)
    @inbounds for j = 1:p
        X[:, j] .= vcol(X, j) .- v[j]
    end
end

"""
    scale(X, v)
Scale each column of `X`.
* `X` : Data.
* `v` : Scaling factors.
""" 
function scale(X, v)
    M = copy(X)
    scale!(M, v)
    M
end

function scale!(X, v)
    p = size(X, 2)
    @inbounds for j = 1:p
        X[:, j] .= vcol(X, j) ./ v[j]
    end
end



