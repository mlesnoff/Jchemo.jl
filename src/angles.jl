"""
    rd(X, Y; typ = :cor)
    rd(X, Y, weights::Weight; typ = :cor)
Compute redundancy coefficients (Rd).
* `X` : Matrix (n, p).
* `Y` : Matrix (n, q).
* `weights` : Weights (n) of the observations. Must be of type `Weight` (see e.g., function `mweight`).
Keyword arguments:
* `typ` : Possibles values are: `:cor` (correlation), `:cov` (uncorrected covariance). 

Returns the redundancy coefficient between `X` and each column of `Y`, i.e. for each k = 1,...,q: 

* Mean {cor(xj, yk)^2 ;  j = 1, ..., p }
    
Depending argument `typ`, the correlation can be replaced by the (not corrected) covariance.

See Tenenhaus 1998 section 2.2.1 p.10-11.

## References
Tenenhaus, M., 1998. La régression PLS: théorie et pratique. Editions Technip, Paris.

## Examples 
```julia 
using Jchemo
X = rand(5, 10)
Y = rand(5, 3)
rd(X, Y)
```
""" 
function rd(X, Y; typ = :cor)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    rd(X, Y, weights; typ)
end

function rd(X, Y, weights::Weight; typ = :cor)
    @assert in([:cor, :cov])(typ) "Wrong value for argument 'typ'." 
    if typ == :cor
        A = corm(X, Y, weights).^2
    elseif typ == :cov
        A = covm(X, Y, weights).^2
    end    
    mean(A; dims = 1)  # keep matrix format
end

"""
    rv(X, Y; centr = true)
    rv(Xbl::Vector; centr = true)
Compute RV coefficients.
* `X` : Matrix (n, p).
* `Y` : Matrix (n, q).
* `Xbl` : A list (vector) of matrices.
* `centr` : Boolean indicating if the matrices will be internally centered or not.

RV is bounded within [0, 1]. 

A dissimilarty measure between `X` and `Y` can be computed by d = sqrt(2 * (1 - RV)).

## References
Escoufier, Y., 1973. Le Traitement des Variables Vectorielles. Biometrics 29, 751–760. 
https://doi.org/10.2307/2529140

Josse, J., Holmes, S., 2016. Measuring multivariate association and beyond. Stat Surv 10, 132–167. 
https://doi.org/10.1214/16-SS116

Josse, J., Pagès, J., Husson, F., 2008. Testing the significance of the RV coefficient. Computational Statistics 
& Data Analysis 53, 82–91. https://doi.org/10.1016/j.csda.2008.06.012

Kazi-Aoual, F., Hitier, S., Sabatier, R., Lebreton, J.-D., 1995. Refined approximations to permutation tests 
for multivariate inference. Computational Statistics & Data Analysis 20, 643–656. 
https://doi.org/10.1016/0167-9473(94)00064-2

Mayer, C.-D., Lorent, J., Horgan, G.W., 2011. Exploratory Analysis of Multiple Omics Datasets Using the Adjusted 
RV Coefficient. Statistical Applications in Genetics and Molecular Biology 10. https://doi.org/10.2202/1544-6115.1540

Smilde, A.K., Kiers, H.A.L., Bijlsma, S., Rubingh, C.M., van Erk, M.J., 2009. Matrix correlations for high-dimensional 
data: the modified RV-coefficient. Bioinformatics 25, 401–405. https://doi.org/10.1093/bioinformatics/btn634

Robert, P., Escoufier, Y., 1976. A Unifying Tool for Linear Multivariate Statistical Methods: The RV-Coefficient. 
Journal of the Royal Statistical Society: Series C (Applied Statistics) 25, 257–265. https://doi.org/10.2307/2347233

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
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    rv(X, Y, weights; centr)
end

function rv(X, Y, weights::Weight; centr = true)
    X = copy(ensure_mat(X))
    Y = copy(ensure_mat(Y))
    n, p = size(X)
    if centr
        fcenter!(X, colmean(X, weights))
        fcenter!(Y, colmean(Y, weights))
    end
    sqrtw = sqrt.(weights.w)
    rweight!(X, sqrtw)
    rweight!(Y, sqrtw)
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
        a = frob2(XtY)
        b = frob2(XtX)
        c = frob2(YtY)
        rv = a / sqrt(b * c)
    end
    rv
end

function rv(Xbl::Vector; centr = true)
    nbl = length(Xbl)
    mat = zeros(eltype(Xbl[1]), nbl, nbl)
    for i = 1:nbl, j = 1:nbl
        mat[i, j] = rv(Xbl[i], Xbl[j]; centr) 
    end
    mat
end



