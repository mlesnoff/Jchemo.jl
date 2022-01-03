"""
    rmgap(X; indexcol, k = 5)
    rmgap!(X; indexcol, k = 5)
Remove vertical gaps in spectra (rows of matrix `X`), e.g. for ASD.  
* `X` : X-data.
* `indexcol` : The indexes of the columns where are located the gaps. 
* `k` : The number of columns used on the left side 
        of the gaps for fitting the linear regressions.

The correction is done by extrapolation from simple linear regressions 
computed on the left side of the gaps. 

For instance, If two gaps are observed between indexes 651-652 and 
between indexes 1425-1426, respectively, then the syntax should 
be `indexcol = [651 ; 1425]`.
""" 
function rmgap(X; indexcol, k = 5)
    rmgap!(copy(X); indexcol, k)
end

function rmgap!(X; indexcol, k = 5)
    X = ensure_mat(X)
    size(X, 2) == 1 ? X = reshape(X, 1, :) : nothing
    p = size(X, 2)
    k = max(k, 2)
    ngap = length(indexcol)
    @inbounds for i = 1:ngap
        ind = indexcol[i]
        wl = max(ind - k + 1, 1):ind
        fm = mlr(Float64.(wl), X[:, wl]')
        pred = predict(fm, ind + 1).pred
        bias = X[:, ind + 1] .- pred'
        X[:, (ind + 1):p] .= X[:, (ind + 1):p] .- bias
    end
    X
end
