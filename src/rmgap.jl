"""
    rmgap(X; kwargs...)
Remove vertical gaps in spectra (e.g. for ASD).  
* `X` : X-data (n, p).
Keyword arguments:
* `indexcol` : Indexes (∈ [1, p]) of the `X`-columns where are 
    located the gaps to remove. 
* `npoint` : The number of `X`-columns used on the left side 
        of each gap for fitting the linear regressions.

For each spectra (row-observation of matrix `X`) and each 
defined gap, the correction is done by extrapolation from 
a simple linear regression computed on the left side of the gap. 

For instance, If two gaps are observed between column-indexes 651-652
and between column-indexes 1425-1426, respectively, the syntax should 
be `indexcol` = [651 ; 1425].

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/asdgap.jld2") 
@load db dat
pnames(dat)
X = dat.X
wlst = names(dat.X)
wl = parse.(Float64, wlst)

wl_target = [1000 ; 1800] 
indexcol = findall(in(wl_target).(wl))

f, ax = plotsp(X, wl)
vlines!(ax, wl_target; linestyle = :dot, color = (:grey, .8))
f

## Corrected data
model = rmgap; indexcol, npoint = 5)
fit!(model, X)
Xc = transf(model, X)
f, ax = plotsp(Xc, wl)
vlines!(ax, wl_target; linestyle = :dot, color = (:grey, .8))
f
```
""" 
function rmgap(X; kwargs...)
    par = recovkw(ParRmgap, kwargs).par
    Rmgap(par)
end

""" 
    transf(object::Rmgap, X)
    transf!(object::Rmgap, X)
Compute the preprocessed data from a model.
* `object` : Model.
* `X` : X-data to transform.
""" 
function transf(object::Rmgap, X)
    X = copy(ensure_mat(X))
    nco(X) == 1 ? X = reshape(X, 1, :) : nothing
    transf!(object, X)
    X
end

function transf!(object::Rmgap, X::Matrix)
    Q = eltype(X)
    p = nco(X)
    indexcol = object.par.indexcol
    npoint = object.par.npoint
    @assert npoint >= 2 "Argument 'npoint' must be >= 2."
    ngap = length(indexcol)
    @inbounds for i = 1:ngap
        ind = indexcol[i]
        wl = max(ind - npoint + 1, 1):ind
        fitm = mlr(convert.(Q, wl), X[:, wl]')
        pred = Jchemo.predict(fitm, ind + 1).pred
        bias = X[:, ind + 1] .- pred'
        X[:, (ind + 1):p] .= X[:, (ind + 1):p] .- bias
    end
end
