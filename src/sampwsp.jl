"""
    sampwsp(X, dcri; maxit = nro(X))
Build training vs. test sets by WSP sampling.  
* `X` : X-data (n, p).
* `dcri` :
Keyword arguments: 
* `maxit` : Maximum number of iterations.

Two outputs (= row indexes of the data) are returned: 
* `train` (`n` - k),
* `test` (k). 

Output `test` is built from the "Wootton, Sergent, Phan-Tan-Luu" (WSP) 
algorithm, assumed to generate samples uniformely distributed in the `X` domain 
(Santiago et al. 2012).

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
n = 600 ; p = 2
X = rand(n, p)
dcri = .5
s = sampwsp(X, dcri)
pnames(res)
@show length(s.test)
plotxy(X[s.test, 1], X[s.test, 2]).f
```
""" 
function sampwsp(X, dcri; maxit = nro(X))
    X = ensure_mat(X)
    n, p = size(X)
    indX = collect(1:n)
    indXtot = copy(indX)
    x = similar(X, 1, p)
    ## First reference point: set as the closest 
    ## from the domain center 
    xmeans = colmean(X)
    s = getknn(X, xmeans'; k = 1).ind[1][1]
    x .= vrow(X, s:s)
    ind = [indX[s]]
    ## Start
    iter = 1
    while (n > 1) && (iter < maxit) 
        res = getknn(X, x; k = n)
        s = res.d[1] .> dcri
        v = res.ind[1][s]  # new {reference point + candidates}
        if length(v) > 0
            s1 = v[1]
            s2 = v[2:end]
            x .= vrow(X, s1:s1)  # reference point
            X = vrow(X, s2)      # candidates
            push!(ind, indX[s1])
            indX = indX[s2]
            n = nro(X)
        else
            train = rmrow(indXtot, ind)
            return (train, test = ind, niter = iter)
        end
        iter += 1
    end
    train = rmrow(indXtot, ind)
    (train, test = ind, niter = iter)
end

