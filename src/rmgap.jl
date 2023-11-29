"""
    rmgap(X; indexcol, k = 5)
    rmgap!(X; indexcol, k = 5)
Remove vertical gaps in spectra , e.g. for ASD.  
* `X` : X-data.
* `indexcol` : The indexes of the columns where are located the gaps. 
* `k` : The number of columns used on the left side 
        of the gaps for fitting the linear regressions.

For each observation (row of matrix `X`),
the corrections are done by extrapolation from simple linear regressions 
computed on the left side of the defined gaps. 

For instance, If two gaps are observed between indexes 651-652 and 
between indexes 1425-1426, respectively, then the syntax should 
be `indexcol = [651 ; 1425]`.

```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/asdgap.jld2") 
@load db dat
pnames(dat)

X = dat.X
wl_str = names(dat.X)
wl = parse.(Float64, wl_str)

z = [1000 ; 1800] 
u = findall(in(z).(wl))
f, ax = plotsp(X, wl)
vlines!(ax, z; linestyle = :dash, color = (:grey, .8))
f

# Corrected data

u = findall(in(z).(wl))
zX = rmgap(X; indexcol = u, k = 5)  
f, ax = plotsp(zX, wl)
vlines!(ax, z; linestyle = :dash, color = (:grey, .8))
f
```
""" 
function rmgap(X; indexcol, k = 5)
    rmgap!(copy(X); indexcol, k)
end

function rmgap!(X; indexcol, k = 5)
    X = ensure_mat(X)
    nco(X) == 1 ? X = reshape(X, 1, :) : nothing
    p = nco(X)
    k = max(k, 2)
    ngap = length(indexcol)
    @inbounds for i = 1:ngap
        ind = indexcol[i]
        wl = max(ind - k + 1, 1):ind
        fm = mlr(Float64.(wl), X[:, wl]')
        pred = Jchemo.predict(fm, ind + 1).pred
        bias = X[:, ind + 1] .- pred'
        X[:, (ind + 1):p] .= X[:, (ind + 1):p] .- bias
    end
    X
end
