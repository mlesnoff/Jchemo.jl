"""
    sampwsp(X, dmin; recod = false, maxit = nro(X))
Build training vs. test sets by WSP sampling.  
* `X` : X-data (n, p).
* `dmin` : Distance "dmin" (Santiago et al. 2012).
Keyword arguments: 
* `recod` : Boolean indicating if `X` is recoded or not 
    before the sampling (see below).
* `maxit` : Maximum number of iterations.

Two outputs (= row indexes of the data) are returned: 
* `train` (`n` - k),
* `test` (k). 

Output `test` is built from the "Wootton, Sergent, Phan-Tan-Luu" (WSP) 
algorithm, assumed to generate samples uniformely distributed in the `X` domain 
(Santiago et al. 2012).

If `recod = true`, each column x of `X` is recoded within [0, 1] and the center of 
the domain is the vector `repeat([.5], p)`. Column x is recoded such as: 
* vmin = minimum(x)
* vmax = maximum(x)
* vdiff = vmax - vmin
* x .=  0.5 .+ (x .- (vdiff / 2 + vmin)) / vdiff

## References

Béal A. 2015. Description et sélection de données en grande dimensio. Thèse de doctorat.
Laboratoire d’Instrumentation et de sciences analytiques,
Ecole doctorale des siences chimiques, Université d'Aix-Marseille.

Santiago, J., Claeys-Bruno, M., Sergent, M., 2012. Construction of space-filling 
designs using WSP algorithm for high dimensional spaces. 
Chemometrics and Intelligent Laboratory Systems, Selected Papers from 
Chimiométrie 2010 113, 26–31. https://doi.org/10.1016/j.chemolab.2011.06.003

## Examples
```julia
using Jchemo

n = 600 ; p = 2
X = rand(n, p)
dmin = .5
s = sampwsp(X, dmin)
pnames(res)
@show length(s.test)
plotxy(X[s.test, 1], X[s.test, 2]).f
```
""" 
function sampwsp(X, dmin; recod = false, maxit = nro(X))
    X = ensure_mat(X)
    n, p = size(X)
    indX = collect(1:n)
    indXtot = copy(indX)
    x = similar(X, 1, p)
    ## First reference point: set as the closest 
    ## from the domain center
    if recod
        zX = recodwsp(X) 
        zX = copy(X)
        xmeans = repeat([.5], p)
    else
        zX = copy(X)
        xmeans = colmean(zX)
        #xmeans = repeat([.5], p)
    end
    s = getknn(zX, xmeans'; k = 1).ind[1][1]
    x .= vrow(zX, s:s)
    ind = [indX[s]]
    ## Start
    iter = 1
    while (n > 1) && (iter < maxit) 
        res = getknn(zX, x; k = n)
        println(res)
        s = res.d[1] .> dmin
        v = res.ind[1][s]  # new {reference point + candidates}
        if length(v) > 0
            s1 = v[1]
            s2 = v[2:end]
            x .= vrow(zX, s1:s1)  # reference point
            zX = vrow(zX, s2)     # candidates
            push!(ind, indX[s1])
            indX = indX[s2]
            n = nro(zX)
        else
            train = rmrow(indXtot, ind)
            return (train, test = ind, niter = iter)
        end
        iter += 1
    end
    train = rmrow(indXtot, ind)
    (train, test = ind, niter = iter)
end

function recodwsp(X)
    X = ensure_mat(X)
    p = nco(X) 
    zX = similar(X)
    for j = 1:p
        x = vcol(X, j)
        vmin = minimum(x)
        vmax = maximum(x)
        vdiff = vmax - vmin
        zX[:, j] .=  0.5 .+ (x .- (vdiff / 2 + vmin)) / vdiff
    end
    zX
end

    
    



