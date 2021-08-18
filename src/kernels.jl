"""
    krbf(X, Y; gamma = 1)
Compute a Radial-Basis-Function (RBF) kernel Gram matrix. 
* `X` : Data.
* `Y` : Data.
* `gamma` : Scale parameter.

Given matrices `X` (n, p) and `Y` (m, p), the function returns
the (n, m) Gram matrix K(X, Y) = Phi(X) * Phi(Y)'.

The RBF kernel between two vectors x and y is computed by 
exp(-gamma * ||x - y||^2).

## References 

Scholkopf, B., Smola, A.J., 2002. Learning with kernels: support vector machines, 
regularization, optimization, and beyond, Adaptive computation and machine learning. 
MIT Press, Cambridge, Mass.

""" 
function krbf(X, Y; gamma = 1)
    exp.(-gamma * euclsq(X, Y))
end

"""
    kpol(X, Y; degree = 1, gamma = 1, coef0 = 0)
Compute a polynomial kernel Gram matrix. 
* `X` : Data.
* `Y` : Data.
* `degree` : Degree of the polynom.
* `gamma` : Scale of the polynom.
* `coef0` : Offset of the polynom.

Given matrices `X` (n, p) and `Y` (m, p), the function returns
the (n, m) Gram matrix K(X, Y) = Phi(X) * Phi(Y)'.

The polynomial kernel between two vectors x and y is computed by 
(gamma * (x' * y) + coef0)^degree.

## References 

Scholkopf, B., Smola, A.J., 2002. Learning with kernels: support vector machines, 
regularization, optimization, and beyond, Adaptive computation and machine learning. 
MIT Press, Cambridge, Mass.

""" 
function kpol(X, Y; degree = 1, gamma = 1, coef0 = 0)
    K = gamma * X * Y' .+ coef0
    if degree > 1
        zK = copy(K)
        @inbounds for i = 1:(degree - 1)
            K .= K .* zK
        end
    end
    K    
end




    