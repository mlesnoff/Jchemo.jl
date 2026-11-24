"""
    outeucl(X; scalx::Symbol = :none)
    outeucl!(X::AbstractMatrix{Q}; scalx::Symbol = :none) where {Q<:Float}
Compute outlierness from Euclidean distances to center.
* `X` : X-data (n, p).
Keyword arguments:
* `scalx` : Symbol defining the column scaling of `X`. Possible values are: `:none` (default), 
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

scalx = :none
#scalx = :mad
res = outeucl(X; scalx) ;
@names res
res.d    # outlierness 
plotxy(1:ntot, res.d).f
```
""" 
function outeucl(X; scalx::Symbol = :none)
    outeucl!(copy(ensure_mat(X)); scalx)
end

function outeucl!(X::AbstractMatrix{Q}; scalx::Symbol = :none) where {Q<:Float}
    centers = Jchemo.colmedspa(X)
    scales = scale(X; scalx)
    if scalx != :none
        fscale!(X, scales)
    end
    csx = (centers = centers, scales)
    d = vec(sqrt.(eucl2(X, centers')))
    (d = d, csx)
end


