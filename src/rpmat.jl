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
function rpmatgauss(p::Int, nlv::Int, Q = Float64)
    randn(Q, p, nlv) / convert(Q, sqrt(nlv))
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
function rpmatli(p::Int, nlv::Int, Q = Float64; 
        spar = sqrt(p))
    le = p * nlv
    k = Int(round(le / spar))
    z = zeros(Q, le)
    u = convert.(Q, [-1 ; 1])
    z[rand(1:le, k)] .= rand(u, k) 
    sparse(reshape(z, p, nlv))
end
