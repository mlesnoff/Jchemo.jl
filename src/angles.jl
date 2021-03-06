"""
    rv(X, Y)
    rv(X_bl)
Compute the RV coefficient between matrices.
* `X` : Matrix (n, p).
* `Y` : Matrix (n, q).
* `X_bl` : A list (vector) of matrices.
* `centr` : Logical indicating if the matrices are internally 
    centered or not.

RV is bounded in [0, 1]. 

A dissimilarty measure between `X` and `Y` can be computed
by d = sqrt(2 * (1 - RV)).

## References
Escoufier, Y., 1973. Le Traitement des Variables Vectorielles. Biometrics 29, 751–760. 
https://doi.org/10.2307/2529140

Josse, J., Holmes, S., 2016. Measuring multivariate association and beyond. 
Stat Surv 10, 132–167. https://doi.org/10.1214/16-SS116

Josse, J., Pagès, J., Husson, F., 2008. Testing the significance of the RV coefficient. 
Computational Statistics & Data Analysis 53, 82–91. https://doi.org/10.1016/j.csda.2008.06.012

Kazi-Aoual, F., Hitier, S., Sabatier, R., Lebreton, J.-D., 1995. Refined approximations 
to permutation tests for multivariate inference. Computational Statistics & Data Analysis 
20, 643–656. https://doi.org/10.1016/0167-9473(94)00064-2

Mayer, C.-D., Lorent, J., Horgan, G.W., 2011. Exploratory Analysis of Multiple Omics 
Datasets Using the Adjusted RV Coefficient. Statistical Applications in Genetics and Molecular
Biology 10. https://doi.org/10.2202/1544-6115.1540

Smilde, A.K., Kiers, H.A.L., Bijlsma, S., Rubingh, C.M., van Erk, M.J., 2009. 
Matrix correlations for high-dimensional data: the modified RV-coefficient. 
Bioinformatics 25, 401–405. https://doi.org/10.1093/bioinformatics/btn634

Robert, P., Escoufier, Y., 1976. A Unifying Tool for Linear Multivariate Statistical Methods: 
The RV-Coefficient. Journal of the Royal Statistical Society: Series C (Applied Statistics) 
25, 257–265. https://doi.org/10.2307/2347233

## Examples 
```julia 
X = rand(5, 10)
Y = rand(5, 3)
rv(X, Y)

X = rand(5, 15) 
listbl = [3:4, 1, [6; 8:10]]
X_bl = mblock(X, listbl)
rv(X_bl)
```
""" 
function rv(X, Y; centr = true)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    if centr
        xmeans = colmean(X)
        ymeans = colmean(Y)
        X = center(X, xmeans)
        Y = center(Y, ymeans)
    end
    if n < p
        XXt = X * X'
        YYt = Y * Y'
        zx = vec(XXt)
        zy = vec(YYt)
        a = zx / norm(zx)
        b = zy / norm(zy)
        rv = a' * b
    else
        XtY = X' * Y    
        XtX = X' * X
        YtY = Y' * Y
        a = ssq(XtY)
        b = ssq(XtX)
        c = ssq(YtY)
        rv = a / sqrt(b * c)
    end
    rv
end

function rv(X_bl::Vector; centr = true)
    nbl = length(X_bl)
    mat = zeros(nbl, nbl)
    for i = 1:(nbl)
        for j = 1:(nbl)
            mat[i, j] = rv(X_bl[i], X_bl[j]; centr = centr)
        end
    end
    mat
end

"""
    lg(X, Y)
    lg(X_bl)
Compute the Lg coefficient between matrices.
* `X` : Matrix (n, p).
* `Y` : Matrix (n, q).
* `X_bl` : A list (vector) of matrices.
* `centr` : Logical indicating if the matrices are internally 
    centered or not.

Lg(X, Y) = Sum_j(=1..p) Sum_k(= 1..q) (cov(xj, yk)^2)

RV(X, Y) = Lg(X, Y) / sqrt(Lg(X, X), Lg(Y, Y))

## References
Escofier, B. & Pagès, J. 1984. L’analyse factorielle multiple. Cahiers du Bureau 
universitaire de recherche opérationnelle. Série Recherche, tome 42, p. 3-68

Escofier, B. & Pagès, J. (2008). Analyses Factorielles Simples et Multiples :
Objectifs, Méthodes et Interprétation. Dunod, 4e édition

## Examples 
```julia 
X = rand(5, 10)
Y = rand(5, 3)
lg(X, Y)

X = rand(5, 15) 
listbl = [3:4, 1, [6; 8:10]]
X_bl = mblock(X, listbl)
lg(X_bl)
```
""" 
function lg(X, Y; centr = true)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = size(X, 1)
    if centr
        xmeans = colmean(X)
        ymeans = colmean(Y)
        X = center(X, xmeans)
        Y = center(Y, ymeans)
    end
    ssq(X' * Y) / n^2 # = sum(cov(X, Y; corrected = false).^2)
end

function lg(X_bl::Vector; centr = true)
    nbl = length(X_bl)
    mat = zeros(nbl, nbl)
    for i = 1:(nbl)
        for j = 1:(nbl)
            mat[i, j] = lg(X_bl[i], X_bl[j]; centr = centr)
        end
    end
    mat
end




