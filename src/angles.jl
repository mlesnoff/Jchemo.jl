"""
    lg(X, Y; centr = true)
    lg(Xbl; centr = true)
Compute the Lg coefficient between matrices.
* `X` : Matrix (n, p).
* `Y` : Matrix (n, q).
* `Xbl` : A list (vector) of matrices.
Keyword arguments:
* `centr` : Boolean indicating if the matrices will 
    be internally centered or not.

Lg(X, Y) = Sum.(j=1..p) Sum.(k= 1..q) cov(xj, yk)^2

RV(X, Y) = Lg(X, Y) / sqrt(Lg(X, X), Lg(Y, Y))

## References
Escofier, B. & Pagès, J. 1984. L’analyse factorielle multiple. 
Cahiers du Bureau universitaire de recherche opérationnelle. 
Série Recherche, tome 42, p. 3-68

Escofier, B. & Pagès, J. (2008). Analyses Factorielles Simples 
et Multiples : Objectifs, Méthodes et Interprétation. Dunod, 
4e édition.

## Examples 
```julia 
using Jchemo
X = rand(5, 10)
Y = rand(5, 3)
lg(X, Y)

X = rand(5, 15) 
listbl = [3:4, 1, [6; 8:10]]
Xbl = mblock(X, listbl)
lg(Xbl)
```
""" 
function lg(X, Y; centr = true)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n = nro(X)
    if centr
        xmeans = colmean(X)
        ymeans = colmean(Y)
        X = fcenter(X, xmeans)
        Y = fcenter(Y, ymeans)
    end
    ## Same as: sum(cov(X, Y; corrected = false).^2)
    ssq(X' * Y) / n^2 
end

function lg(Xbl::Vector; centr = true)
    nbl = length(Xbl)
    mat = zeros(eltype(Xbl[1]), nbl, nbl)
    for i = 1:nbl
        for j = 1:nbl
            mat[i, j] = lg(Xbl[i], Xbl[j]; centr)
        end
    end
    mat
end

"""
    rd(X, Y; typ = :cor)
    rd(X, Y, weights::Weight; typ = :cor)
Compute redundancy coefficients between two matrices.
* `X` : Matrix (n, p).
* `Y` : Matrix (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `typ` : Possibles values are: `:cor` (correlation), 
    `:cov` (uncorrected covariance). 

Returns the redundancy coefficient between `X` and each column
of `Y`, i.e.: 

(1 / p) * [Sum.(j=1, .., p) cor(xj, y1)^2 ; ... ; Sum.(j=1, .., p) cor(xj, yq)^2] 

See Tenenhaus 1998 section 2.2.1 p.10-11.

## References
Tenenhaus, M., 1998. La régression PLS: théorie et pratique. 
Editions Technip, Paris.

## Examples 
```julia 
using Jchemo
X = rand(5, 10)
Y = rand(5, 3)
rd(X, Y)
```
""" 
function rd(X, Y; typ = :cor)
    @assert in([:cor, :cov])(typ) "Wrong value for argument 'typ'." 
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    p = nco(X)
    if typ == :cor
        A = cor(X, Y).^2
    elseif typ == :cov
        A = cov(X, Y; corrected = false).^2
    end    
    sum(A; dims = 1) / p
end

function rd(X, Y, weights::Weight; typ = :cor)
    @assert in([:cor, :cov])(typ) "Wrong value for argument 'typ'." 
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    p = nco(X)
    typ == :cor ? algo = corm : algo = covm
    A = algo(X, Y, weights).^2
    sum(A; dims = 1) / p
end

"""
    rv(X, Y; centr = true)
    rv(Xbl::Vector; centr = true)
Compute the RV coefficient between matrices.
* `X` : Matrix (n, p).
* `Y` : Matrix (n, q).
* `Xbl` : A list (vector) of matrices.
* `centr` : Boolean indicating if the matrices 
    will be internally centered or not.

RV is bounded in [0, 1]. 

A dissimilarty measure between `X` and `Y` can be computed
by d = sqrt(2 * (1 - RV)).

## References
Escoufier, Y., 1973. Le Traitement des Variables Vectorielles. 
Biometrics 29, 751–760. https://doi.org/10.2307/2529140

Josse, J., Holmes, S., 2016. Measuring multivariate association and beyond. 
Stat Surv 10, 132–167. https://doi.org/10.1214/16-SS116

Josse, J., Pagès, J., Husson, F., 2008. Testing the significance of 
the RV coefficient. Computational Statistics & Data Analysis 53, 82–91. 
https://doi.org/10.1016/j.csda.2008.06.012

Kazi-Aoual, F., Hitier, S., Sabatier, R., Lebreton, J.-D., 1995. 
Refined approximations to permutation tests for multivariate inference. 
Computational Statistics & Data Analysis 20, 643–656. 
https://doi.org/10.1016/0167-9473(94)00064-2

Mayer, C.-D., Lorent, J., Horgan, G.W., 2011. Exploratory Analysis 
of Multiple Omics Datasets Using the Adjusted RV Coefficient. Statistical 
Applications in Genetics and Molecular Biology 10. https://doi.org/10.2202/1544-6115.1540

Smilde, A.K., Kiers, H.A.L., Bijlsma, S., Rubingh, C.M., van Erk, M.J., 2009. 
Matrix correlations for high-dimensional data: the modified RV-coefficient. 
Bioinformatics 25, 401–405. https://doi.org/10.1093/bioinformatics/btn634

Robert, P., Escoufier, Y., 1976. A Unifying Tool for Linear Multivariate 
Statistical Methods: The RV-Coefficient. Journal of the Royal Statistical Society: 
Series C (Applied Statistics) 25, 257–265. https://doi.org/10.2307/2347233

## Examples 
```julia 
using Jchemo
X = rand(5, 10)
Y = rand(5, 3)
rv(X, Y)

X = rand(5, 15) 
listbl = [3:4, 1, [6; 8:10]]
Xbl = mblock(X, listbl)
rv(Xbl)
```
""" 
function rv(X, Y; centr = true)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    n, p = size(X)
    if centr
        xmeans = colmean(X)
        ymeans = colmean(Y)
        X = fcenter(X, xmeans)
        Y = fcenter(Y, ymeans)
    end
    if n < p
        XXt = X * X'
        YYt = Y * Y'
        zx = vec(XXt)
        zy = vec(YYt)
        a = zx / normv(zx)
        b = zy / normv(zy)
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

function rv(Xbl::Vector; centr = true)
    nbl = length(Xbl)
    mat = zeros(eltype(Xbl[1]), nbl, nbl)
    for i = 1:nbl
        for j = 1:nbl
            mat[i, j] = rv(Xbl[i], Xbl[j]; centr)
        end
    end
    mat
end

