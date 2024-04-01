"""
    fdasvd(; kwargs...)
    fdasvd(X, y; kwargs...)
    fdasvd!(X::Matrix, y; kwargs...)
Factorial discriminant analysis (FDA).
* `X` : X-data (n, p).
* `y` : y-data (n) (class membership).
Keyword arguments:
* `nlv` : Nb. of discriminant components.
* `lb` : Ridge regularization parameter "lambda".
    Can be used when `X` has collinearities. 
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

FDA by a weighted SVD factorization of the matrix of the 
class centers (after spherical transformaton). 
The function gives the same results as function `fda`.

See function `fda` for examples.
""" 
fdasvd(X, y; kwargs...) = fdasvd!(copy(ensure_mat(X)), y; kwargs...)

function fdasvd!(X::Matrix, y; kwargs...)
    par = recovkwargs(Par, kwargs)
    @assert par.lb >= 0 "Argument 'lb' must ∈ [0, Inf[."
    Q = eltype(X)
    n, p = size(X)
    lb = convert(Q, par.lb)
    xmeans = colmean(X) 
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    w = mweight(ones(Q, n))
    res = matW(X, y, w)
    lev = res.lev
    nlev = length(lev)
    ni = res.ni
    res.W .*= n / (n - nlev)
    if lb > 0
        res.W .= res.W .+ lb .* I(p) # @. does not work with I
    end
    #Winv = inv(res.W)
    Winv = LinearAlgebra.inv!(cholesky(Hermitian(res.W))) 
    ct = aggstat(X, y; fun = mean).X
    Ut = cholesky!(Hermitian(Winv)).U'
    Zct = ct * Ut
    nlv = min(par.nlv, n, p, nlev - 1)
    par = Par(nlv = nlv, scal = false)
    weights = mweight(convert.(Q, ni))
    fm = pcasvd(Zct, weights; kwargs...)
    Pz = fm.P
    Tcenters = Zct * Pz
    eig = (fm.sv).^2 
    sstot = sum(eig)
    P = Ut * Pz[:, 1:nlv]
    T = X * P
    Tcenters = ct * P
    Fda(T, P, Tcenters, eig, sstot, res.W, xmeans, 
        xscales, lev, ni, kwargs, par)
end


