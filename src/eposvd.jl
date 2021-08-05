"""
    eposvd(D; nlv)
External parameter orthogonalization (EPO).
* `D` : A dataset representing detrimental information.
* `nlv` : The number of first loadings vectors of `D` 
    considered for the orthogonalization.

The objective of EPO method (Roger et al 2003)
is to remove from a dataset X (n, p) some detrimental information 
(e.g. humidity effect) represented by a dataset `D` (m, p).

EPO consists in orthogonalizing the row observations of X to 
the detrimental sub-space defined by the first `nlv` non-centered 
PCA loading vectors of `D`.
        
Function `eposvd` uses a SVD factorization of `D`, and returns
an orthogonalization matrix `M` (p, p) and a matrix `P` whose  
columns are the `nlv` loading vectors of `D`.
        
The EPO correction from `D` consists in:  X_corrected = X * M.

## References

Roger, J.-M., Chauchard, F., Bellon-Maurel, V., 2003. EPO-PLS external parameter 
orthogonalisation of PLS application to temperature-independent measurement 
of sugar content of intact fruits. Chemometrics and Intelligent Laboratory Systems 66, 191-204. 
https://doi.org/10.1016/S0169-7439(03)00051-0

Roger, J.-M., Boulet, J.-C., 2018. A review of orthogonal projections for calibration. 
Journal of Chemometrics 32, e3045. https://doi.org/10.1002/cem.3045

""" 
function eposvd(D; nlv)
    D = ensure_mat(D)
    m, p = size(D)
    nlv = min(nlv, m, p)
    I = Diagonal(ones(p))
    if nlv == 0 
        M = I
        P = nothing
    else 
        P = svd(D).V[:, 1:nlv]
        M = I - P * P'
    end
    (M = M, P = P)
end

