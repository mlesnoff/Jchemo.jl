"""
    fdasvd(X, y, weights; kwargs...)
    fdasvd!(X::Matrix, y, weights; kwargs...)
Factorial discriminant analysis (FDA).
* `X` : X-data (n, p).
* `y` : y-data (n) (class membership).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `nlv` : Nb. of discriminant components.
* `lb` : Ridge regularization parameter "lambda".
    Can be used when `X` has collinearities. 
* `prior` : Type of prior probabilities for class 
    membership. Possible values are: `:unif` (uniform), 
    `:prop` (proportional), or a vector (of length equal to 
    the number of classes) giving the prior weight for each class 
    (the vector must be sorted in the same order as `mlev(y)`).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

FDA by a weighted SVD factorization of the matrix of the 
class centers (after spherical transformaton). 
The function gives the same results as function `fda`.

See function `fda` for details and examples.
""" 
function fdasvd(X, y; kwargs...)
    par = recovkw(ParFda, kwargs).par
    Q = eltype(X[1, 1])
    weights = mweightcla(Q, y; prior = par.prior)
    fdasvd(X, y, weights; kwargs...)
end

fdasvd(X, y, weights; kwargs...) = fda!(copy(ensure_mat(X)), y, weights; kwargs...)

function fdasvd!(X::Matrix, y, weights; kwargs...)
    par = recovkw(Par, kwargs).par
    @assert par.lb >= 0 "Argument 'lb' must ∈ [0, Inf[."
    Q = eltype(X)
    n, p = size(X)
    lb = convert(Q, par.lb)
    xmeans = colmean(X, weights)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        fcscale!(X, xmeans, xscales)
    else
        fcenter!(X, xmeans)
    end
    res = matW(X, y, weights)
    lev = res.lev
    nlev = length(lev)
    ni = res.ni
    res.W .*= n / (n - nlev)
    if lb > 0
        res.W .+= lb .* I(p) # @. does not work with I
    end
    #Winv = inv(res.W)
    Winv = LinearAlgebra.inv!(cholesky(Hermitian(res.W))) 
    ct = similar(X, nlev, p)
    @inbounds for i in eachindex(lev)
        s = findall(y .== lev[i]) 
        ct[i, :] = colmean(vrow(X, s), mweight(weights.w[s]))
    end
    #ct = aggstat(X, y; fun = mean).X
    Ut = cholesky!(Hermitian(Winv)).U'
    Zct = ct * Ut
    nlv = min(par.nlv, n, p, nlev - 1)
    par = Par(nlv = nlv, scal = false)
    zweights = mweight(convert.(Q, ni))
    fm = pcasvd(Zct, zweights; kwargs...)
    Pz = fm.P
    Tcenters = Zct * Pz
    eig = (fm.sv).^2 
    sstot = sum(eig)
    P = Ut * Pz[:, 1:nlv]
    T = X * P
    Tcenters = ct * P
    Fda(T, P, Tcenters, eig, sstot, res.W, xmeans, xscales, weights, 
        lev, ni, par)
end


