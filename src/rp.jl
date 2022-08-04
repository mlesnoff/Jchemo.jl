struct Rp
    T::Matrix{Float64}
    P
end

"""
    rpmat_gauss(p, a)
Build a gaussian random projection matrix.
* `p` : Nb. variables (attributes) to project.
* `a` : Nb. of simulated projection dimensions.

The function returns a random projection matrix P of dimension 
`p` x `a`. The projection of a given matrix X of size n x `p` is given
by X * P.

P is simulated from i.i.d. N(0, 1)/sqrt(`a`).

## References 
Li, P., Hastie, T.J., Church, K.W., 2006. Very sparse random projections, 
in: Proceedings of the 12th ACM SIGKDD International Conference on Knowledge 
Discovery and Data Mining, KDD ’06. Association for Computing Machinery,
New York, NY, USA, pp. 287–296. https://doi.org/10.1145/1150402.1150436

## Examples
```julia
p = 10 ; a = 3
rpmat_gauss(p, a)
```
""" 
function rpmat_gauss(p, a)
    randn(p, a) / sqrt(a)
end


"""
    rpmat_li(p, a; s = sqrt(p))
Build a sparse random projection matrix (Achlioptas 2001, Li et al. 2006).
* `p` : Nb. variables (attributes) to project.
* `a` : Nb. final dimensions, i.e. after projection.
* `s` : Coefficient defining the sparsity of the returned matrix 
    (higher is `s`, higher is the sparsity).


The function returns a random projection matrix P of dimension 
`p` x `a`. The projection of a given matrix X of size n x `p` is given
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
p = 10 ; a = 3
rpmat_li(p, a)
```
""" 
function rpmat_li(p, a; s = sqrt(p))
    le = p * a
    k = Int64(round(le / s))
    z = zeros(le)
    z[rand(1:le, k)] .= rand([-1. ; 1], k) 
    sparse(reshape(z, p, a))
end

"""
    rp(X; a, fun = rpmat_li, kwargs ...)
Make a random projection of matrix X.
* `X` : X-data to project.
* `a` : Nb. dimensions on which `X` is projected.
* `fun` : A function of random projection.
* `kwargs` : Optional arguments of function `fun`.

## Examples
```julia
X = rand(5, 10)
a = 3
fm = rp(X; a = a)
pnames(fm)
size(fm.P) 
fm.P
fm.T # = X * fm.P 
transform(fm, X[1:2, :])
```
""" 
function rp(X; a, fun = rpmat_li, kwargs ...)
    X = ensure_mat(X)
    P = fun(size(X, 2), a; kwargs...)
    T = X * P
    Rp(T, P)
end

""" 
    transform(object::Rp, X; a = nothing)
Compute "scores" T from a random projection model and a matrix X.
* `object` : The random projection model.
* `X` : Matrix (m, p) for which LVs are computed.
* `a` : Nb. dimensions to consider. If nothing, it is the maximum nb. dimensions.
""" 
function transform(object::Rp, X; a = nothing)
    a = size(object.T, 2)
    isnothing(a) ? a = a : a = min(a, a)
    X * vcol(object.P, 1:a)
end

