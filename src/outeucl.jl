"""
    outeucl(X; scal::Symbol = :none)
    outeucl!(X::AbstractMatrix{Q}; scal::Symbol = :none) where {Q<:Float}
Compute outlierness from Euclidean distances to center.
* `X` : X-data (n, p).
Keyword arguments:
* `scal` : Symbol defining the column scaling of `X`. Possible values are: `:none` (default), 
    `:std` (uncorrected STD), `:prt` (Pareto), and `:mad` (MAD).

This function computes outlierness `d` of each observation (row) of `X` by the Euclidean distance between the observation and a 
robust estimate of the center of the data (in the present function, the spatial median). Such outlierness was for instance 
used in the robust PLSR algorithm 'PRM' of Serneels et al. 2005. 

# References
Serneels, S., Croux, C., Filzmoser, V., Van Espen, V.J., 2005. Partial robust M-regression. 
Chemometrics and Intelligent Laboratory Systems 79, 55-64. https://doi.org/10.1016/j.chemolab.2005.04.007

# Examples
```julia
using Jchemo, CairoMakie
n = 300 ; p = 700 ; m = 80
ntot = n + m
X1 = randn(n, p)
X2 = randn(m, p) .+ rand(1:3, p)'
X = vcat(X1, X2)

scal = :none
#scal = :mad
res = outeucl(X; scal) ;
@names res
res.d    # outlierness 
plotxy(1:ntot, res.d).f
```
""" 
function outeucl(X; scal::Symbol = :none)
    outeucl!(copy(ensure_mat(X)); scal)
end

function outeucl!(X::AbstractMatrix{Q}; scal::Symbol = :none) where {Q<:Float}
    centers = Jchemo.colmedspa(X)
    scales = scale(X; scal = par.scal)
    
    if scal != :none
        fscale!(X, xscales)
    end

    cs = cscale(X, weights; centr = :mean, scal = par.scal)
    transf!(cs, X)

    d = vec(sqrt.(eucl2(X, xcenters')))
    (d = d, cs)
end


