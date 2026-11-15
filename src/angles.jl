"""
    rd(X::AbstMatVec{Q}, Y::AbstMatVec{Q}; typ::Symbol = :cor) where {Q<:Float}
    rd(X::AbstMatVec{Q}, Y::AbstMatVec{Q}, weights::ProbabilityWeights{Q}; 
        typ::Symbol = :cor) where {Q<:Float}
Compute redundancy coefficients (Rd).
* `X` : Matrix (n, p).
* `Y` : Matrix (n, q).
* `weights` : Weights (n) of the observations. Must be of type `ProbabilityWeights` (see e.g., function `pweight`).
Keyword arguments:
* `typ` : Possibles values are: `:cor` (correlation), `:cov` (uncorrected covariance). 

Returns the redundancy coefficient between `X` and each column of `Y`, i.e. for each k = 1,...,q: 

* Mean {cor(xj, yk)^2 ;  j = 1, ..., p }
    
Depending argument `typ`, the correlation can be replaced by the (not corrected) covariance.

See Tenenhaus 1998 section 2.2.1 p.10-11.

# References
Tenenhaus, M., 1998. La régression PLS: théorie et pratique. Editions Technip, Paris.

# Examples 
```julia 
using Jchemo
X = rand(5, 10)
Y = rand(5, 3)
rd(X, Y)
```
""" 
function rd(X::AbstMatVec{Q}, Y::AbstMatVec{Q}; typ::Symbol = :cor) where {Q<:Float}
    weights = pweight(ones(eltype(X), nro(X)))
    rd(X, Y, weights; typ)
end

function rd(X::AbstMatVec{Q}, Y::AbstMatVec{Q}, weights::ProbabilityWeights{Q}; 
        typ::Symbol = :cor) where {Q<:Float}
    @assert @in(typ, [:cor, :cov]) "Wrong value for argument 'typ'." 
    if typ == :cor
        A = corm(X, Y, weights).^2
    elseif typ == :cov
        A = covm(X, Y, weights).^2
    end    
    mean(A; dims = 1)  # use 'mean' since keep matrix format
end

"""
    rv(X::AbstMatVec{Q}, Y::AbstMatVec{Q}) where {Q<:Float}
    rv(X::AbstMatVec{Q}, Y::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where {Q<:Float}
    rv(Xbl)
    rv(Xbl::Vector{Matrix{Q}}, weights::ProbabilityWeights{Q}) where {Q<:Float}
Compute RV coefficients.
* `X` : Matrix (n, p) or vector (n).
* `Y` : Matrix (n, q) or vector (n).
* `Xbl` : A list (vector) of matrices.

`X` and `Y` (or each block in `Xbl`) are internally centered before RV computation.

* RV is bounded within [0, 1]. 

* A dissimilarty measure between two matrices X and Y can be computed by d = sqrt(2 * (1 - RV(X, Y))).

# References
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

# Examples 
```julia 
using Jchemo

X = [1. 2 3 4 5 7 100; 4 1 6 7 12 13 28; 12 5 6 13 3 1 5; 27 18 7 6 2 0 12; 
    12 11 28 7 1 25 2; 2 3 7 1 0 7 26; 14 12 101 4 3 7 10; 8 7 6 5 4 3 2] 
Y = [10. 11; 120 131; 8 12; 1 2; 100 10; 1 27; 8 18; 1.3 130] 
listbl = [1:3, 4, 5:7]
Xbl = mblock(X, listbl) 

rv(X, Y)         # 0.072634
rv(X, Y[:, 1])   # 0.039809

rv(Xbl)
```
""" 
function rv(X::AbstMatVec{Q}, Y::AbstMatVec{Q}) where {Q<:Float}
    weights = pweight(ones(eltype(X), nro(X)))
    rv(X, Y, weights)
end

function rv(X::AbstMatVec{Q}, Y::AbstMatVec{Q}, weights::ProbabilityWeights{Q}) where {Q<:Float}
    X = copy(ensure_mat(X))
    Y = copy(ensure_mat(Y))
    n, p = size(X)
    # Centering
    fitmx = center(X, weights; typ = :mean)
    transf!(fitmx, X)
    fitmy = center(Y, weights; typ = :mean)
    transf!(fitmy, Y)
    # End
    sqrtw = sqrt.(weights.values)
    fweightr!(X, sqrtw)
    fweightr!(Y, sqrtw)
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

function rv(Xbl)
    Xbl = ensure_mat_mb(Xbl)
    weights = pweight(ones(eltype(Xbl[1]), nro(Xbl[1])))
    rv(Xbl, weights)
end

function rv(Xbl::Vector{Matrix{Q}}, weights::ProbabilityWeights{Q}) where {Q<:Float}
    nbl = length(Xbl)
    mat = Array{Q}(undef, nbl, nbl)
    for i in eachindex(Xbl), j in eachindex(Xbl)
        mat[i, j] = rv(Xbl[i], Xbl[j], weights) 
    end
    mat
end


