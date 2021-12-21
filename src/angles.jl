"""
    rv(X, Y)
Compute the sample RV coefficient between matrices X and Y
* `X` : Matrix (n obs., p variables).
* `Y` : Matrix (n obs., q variables).

RV is bounded in [0, 1]. 

A dissimilarty measure between `X` and `Y` can be computed
by d = sqrt(2 * (1 - RV)).


## References

Escoufier, Y., 1973. Le Traitement des Variables Vectorielles. Biometrics 29, 751–760. 
https://doi.org/10.2307/2529140

Josse, J., Pagès, J., Husson, F., 2008. Testing the significance of the RV coefficient. 
Computational Statistics & Data Analysis 53, 82–91. https://doi.org/10.1016/j.csda.2008.06.012

Mayer, C.-D., Lorent, J., Horgan, G.W., 2011. Exploratory Analysis of Multiple Omics 
Datasets Using the Adjusted RV Coefficient. Statistical Applications in Genetics and Molecular
Biology 10. https://doi.org/10.2202/1544-6115.1540

Smilde, A.K., Kiers, H.A.L., Bijlsma, S., Rubingh, C.M., van Erk, M.J., 2009. 
Matrix correlations for high-dimensional data: the modified RV-coefficient. 
Bioinformatics 25, 401–405. https://doi.org/10.1093/bioinformatics/btn634

Robert, P., Escoufier, Y., 1976. A Unifying Tool for Linear Multivariate Statistical Methods: 
The RV-Coefficient. Journal of the Royal Statistical Society: Series C (Applied Statistics) 
25, 257–265. https://doi.org/10.2307/2347233
""" 

function rv(X, Y; centr = true)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    if centr
        X = center(X, mean(X, dims = 1))
        Y = center(Y, mean(Y, dims = 1))
    end
    YtX = Y' * X    
    XtX = X' * X
    YtY = Y' * Y
    A = dot(YtX, YtX)
    B = dot(XtX, XtX)
    C = dot(YtY, YtY)
    rv = A / sqrt(B * C)
end

