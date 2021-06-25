"""
    krbf(X, Y, sigma = 1)
Compute the kernel Gram matrix for Radial-Basis-Function (RBF)
* `X` : matrix (n, p), or vector (n,).
* `Y` : matrix (m, p), or vector (m,).
* `sigma` : scale parameter.

The RBF kernel between two vectors *x* and *y* is defined by *exp(-.5 * (x - y)^2 / sigma^2)*.

The function returns the Gram matrix K(X, Y) = Phi(X) * Phi(Y)' of size (n, m).

## References 

Scholkopf, B., Smola, A.J., 2002. Learning with kernels: support vector machines, 
regularization, optimization, and beyond, Adaptive computation and machine learning. 
MIT Press, Cambridge, Mass.

""" 
function krbf(X, Y; sigma = 1)
    exp.(-.5 * euclsq(X, Y) / sigma^2)
end

function kpol(X, Y; degree = 1, scale = 1, offset = 0)
    K = scale * X * Y' .+ offset
    if degree > 1
        zK = copy(K)
        @inbounds for i = 1:(degree - 1)
            K .= K .* zK
        end
    end
    K    
end




    