"""
    difmean(X1, X2)
Compute a detrimental matrix (for calibration transfer) by column means difference.
* `X1` : Matrix of spectra (n1, p).
* `X2` : Matrix of spectra (n2, p).
* `normx` : Boolean, if * `true`, the vectors of the column means 
    of `X1` and `X2` are normed before computing the difference.

The function returns a matrix D (1, p) containing the detrimental information
that has to be removed from spectra `X1` and `X2` for calibration transfer 
by orthogonalization (e.g. input for function `eposvd`). Matrix D is computed 
by the difference between the two mean spectra (column means of `X1` and `X2`).

## Examples
```julia
```
"""
function difmean(X1, X2; normx::Bool = false)
    xmeans1 = colmean(X1)
    xmeans2 = colmean(X2)
    if normx
        xmeans1 ./= norm(xmeans1)
        xmeans2 ./= norm(xmeans2)
    end
    D = (xmeans1 - xmeans2)'
    (D = D, xmeans1, xmeans2)
end
