"""
    krbf(X, Y, sigma = 1)
Compute a Radial-Basis-Function (RBF) kernel Gram matrix. 
* `X` : Data.
* `Y` : Data.
* `sigma` : scale parameter.

When `X` (n, p) and `Y` (m, p), it returns
the (n, m) Gram matrix K(X, Y) = Phi(X) * Phi(Y)'.

The RBF kernel between two vectors x and y is defined by exp(-.5 * (x - y)^2 / sigma^2).

## References 

Scholkopf, B., Smola, A.J., 2002. Learning with kernels: support vector machines, 
regularization, optimization, and beyond, Adaptive computation and machine learning. 
MIT Press, Cambridge, Mass.

""" 
function krbf(X, Y; sigma = 1)
    exp.(-.5 * euclsq(X, Y) / sigma^2)
end

"""
    kpol(X, Y; degree = 1, scale = 1, offset = 0)
Compute a polynomial kernel Gram matrix. 
* `X` : Data.
* `Y` : Data.
* `degree` : degree of the polynom.
* `scale` : Scale of the polynom.
* `offset` : Offset of the polynom.

When `X` (n, p) and `Y` (m, p), it returns
the (n, m) Gram matrix K(X, Y) = Phi(X) * Phi(Y)'.

The polynomial kernel between two vectors x and y is defined by (scale * (x'y) + offset)^degree.

## References 

Scholkopf, B., Smola, A.J., 2002. Learning with kernels: support vector machines, 
regularization, optimization, and beyond, Adaptive computation and machine learning. 
MIT Press, Cambridge, Mass.

""" 
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




    