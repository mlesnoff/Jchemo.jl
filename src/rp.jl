struct Rp
    T::Matrix{Float64}
    P
    xmeans
    xscales
end

"""
    rpmatgauss(p, nlv)
Build a gaussian random projection matrix.
* `p` : Nb. variables (attributes) to project.
* `nlv` : Nb. of simulated projection dimensions.

The function returns a random projection matrix P of dimension 
`p` x `nlv`. The projection of a given matrix X of size n x `p` is given
by X * P.

P is simulated from i.i.d. N(0, 1)/sqrt(`nlv`).

## References 
Li, P., Hastie, T.J., Church, K.W., 2006. Very sparse random projections, 
in: Proceedings of the 12th ACM SIGKDD International Conference on Knowledge 
Discovery and Data Mining, KDD ’06. Association for Computing Machinery,
New York, NY, USA, pp. 287–296. https://doi.org/10.1145/1150402.1150436

## Examples
```julia
p = 10 ; nlv = 3
rpmatgauss(p, nlv)
```
""" 
function rpmatgauss(p, nlv)
    randn(p, nlv) / sqrt(nlv)
end


"""
    rpmatli(p, nlv; s = sqrt(p))
Build a sparse random projection matrix (Achlioptas 2001, Li et al. 2006).
* `p` : Nb. variables (attributes) to project.
* `nlv` : Nb. final dimensions, i.e. after projection.
* `s` : Coefficient defining the sparsity of the returned matrix 
    (higher is `s`, higher is the sparsity).


The function returns a random projection matrix P of dimension 
`p` x `nlv`. The projection of a given matrix X of size n x `p` is given
by X * P.

P is simulated from i.i.d. "p_ij" = 
* 1 with prob. 1/(2 * `s`)
* 0 with prob. 1 - 1 / `s`
* -1 with prob. 1/(2 * `s`)

Usual values for `s` are:
* sqrt(`p`)       (Li et al. 2006)
* `p` / log(`p`)  (Li et al. 2006)
* 1               (Achlioptas 2001)
* 3               (Achlioptas 2001) 

## References 
Achlioptas, D., 2001. Database-friendly random projections, 
in: Proceedings of the Twentieth ACM SIGMOD-SIGACT-SIGART Symposium on 
Principles of Database Systems, PODS ’01. Association for Computing Machinery, 
New York, NY, USA, pp. 274–281. https://doi.org/10.1145/375551.375608

Li, P., Hastie, T.J., Church, K.W., 2006. Very sparse random projections, 
in: Proceedings of the 12th ACM SIGKDD International Conference on Knowledge 
Discovery and Data Mining, KDD ’06. Association for Computing Machinery,
New York, NY, USA, pp. 287–296. https://doi.org/10.1145/1150402.1150436

## Examples
```julia
p = 10 ; nlv = 3
rpmatli(p, nlv)
```
""" 
function rpmatli(p, nlv; s = sqrt(p))
    le = p * nlv
    k = Int64(round(le / s))
    z = zeros(le)
    z[rand(1:le, k)] .= rand([-1. ; 1], k) 
    sparse(reshape(z, p, nlv))
end

"""
    rp(X, weights = ones(nro(X)); nlv, fun = rpmatli, scal::Bool = false, kwargs ...)
    rp!(X::Matrix, weights = ones(nro(X)); nlv, fun = rpmatli, scal::Bool = false, kwargs ...)
Make a random projection of matrix X.
* `X` : X-data (n, p).
* `weights` : Weights (n) of the observations. Internally normalized to sum to 1.
* `nlv` : Nb. dimensions on which `X` is projected.
* `fun` : A function of random projection.
* `kwargs` : Optional arguments of function `fun`.
* `scal` : Boolean. If `true`, each column of `X` is scaled
    by its uncorrected standard deviation.

## Examples
```julia
X = rand(5, 10)
nlv = 3
fm = rp(X; nlv = nlv)
pnames(fm)
size(fm.P) 
fm.P
fm.T # = X * fm.P 
Jchemo.transform(fm, X[1:2, :])
```
"""
function rp(X, weights = ones(nro(X)); nlv, fun = rpmatli, 
    scal::Bool = false, kwargs...)
    rp!(copy(ensure_mat(X)), weights; nlv, 
        fun = fun, scal = scal, kwargs...)
end

function rp!(X::Matrix, weights = ones(nro(X)); nlv, fun = rpmatli, 
        scal::Bool = false, kwargs...)
    X = ensure_mat(X)
    p = nco(X)
    weights = mweight(weights)
    xmeans = colmean(X, weights)
    xscales = ones(p)
    if scal 
        xscales .= colstd(X, weights)
        cscale!(X, xmeans, xscales)
    else
        center!(X, xmeans)
    end
    P = fun(p, nlv; kwargs...)
    T = X * P
    Rp(T, P, xmeans, xscales)
end

""" 
    transform(object::Rp, X; nlv = nothing)
Compute "scores" T from a random projection model and a matrix X.
* `object` : The random projection model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. dimensions to consider. If nothing, it is the maximum nb. dimensions.
""" 
function transform(object::Rp, X; nlv = nothing)
    X = ensure_mat(X)
    a = size(object.T, 2)
    isnothing(nlv) ? nlv = a : nlv = min(nlv, a)
    cscale(X, object.xmeans, object.xscales) * vcol(object.P, 1:nlv)
end

