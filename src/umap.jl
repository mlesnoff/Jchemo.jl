"""
    umap(X, Y; kwargs...)
UMAP: Uniform manifold approximation and projection for 
    dimension reduction
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
* `n_neighbors` : Nb. approximate neighbors used to construct 
    the initial high-dimensional graph.
* `min_dist` : Minimum distance between points in low-dimensional 
    space.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.
    
The function fits a UMAP dimension reducion using 
package `UMAP.jl'. The used metric is the Euclidean distance. 

## References

https://github.com/dillondaudert/UMAP.jl

McInnes, L, Healy, J, Melville, J, UMAP: Uniform Manifold Approximation 
and Projection for Dimension Reduction. ArXiV 1802.03426, 2018
https://arxiv.org/abs/1802.03426

https://umap-learn.readthedocs.io/en/latest/how_umap_works.html

https://pair-code.github.io/understanding-umap/ 

## Examples
```julia
using Jchemo, JchemoData
using JLD2, GLMakie, CairoMakie, FreqTables
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "challenge2018.jld2") 
@load db dat
pnames(dat)
X = dat.X 
Y = dat.Y
wlst = names(X)
wl = parse.(Float64, wlst)
ntot = nro(X)
summ(Y)
typ = Y.typ
test = Y.test
y = Y.conc

mod1 = model(snv; centr = true, scal = true) 
mod2 = model(savgol; npoint = 21, deriv = 2, degree = 3)
mod = pip(mod1, mod2)
fit!(mod, X)
@head Xp = transf(mod, X)
plotsp(Xp, wl; xlabel = "Wavelength (nm)", ylabel = "Absorbance", nsamp = 20).f

s = Bool.(test)
Xtrain = rmrow(Xp, s)
Ytrain = rmrow(Y, s)
ytrain = rmrow(y, s)
typtrain = rmrow(typ, s)
Xtest = Xp[s, :]
Ytest = Y[s, :]
ytest = y[s]
typtest = typ[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
(ntot = ntot, ntrain, ntest)

freqtable(string.(typ, "-", Y.label))
freqtable(typ, test)

#################

nlv = 3
n_neighbors = 50 ; min_dist = .5 
mod = model(umap; nlv, n_neighbors, min_dist)  
fit!(mod, Xtrain)
@head T = mod.fm.T
@head Ttest = transf(mod, Xtest)

GLMakie.activate!() 
#CairoMakie.activate!()
ztyp = recod_catbyint(typtrain)
colsh = :tab10
f = Figure()
i = 1
ax = Axis3(f[1, 1], xlabel = string("LV", i), ylabel = string("LV", i + 1), 
        zlabel = string("LV", i + 2), title = "UMAP", perspectiveness = 0) 
scatter!(ax, T[:, i], T[:, i + 1], T[:, i + 2]; markersize = 8, 
    color = ztyp, colormap = colsh) 
scatter!(ax, Ttest[:, i], Ttest[:, i + 1], Ttest[:, i + 2], color = :black, 
    colormap = :tab20, markersize = 10)  
lev = mlev(typtrain)
nlev = length(lev)
colm = cgrad(colsh, nlev; alpha = .7, categorical = true) 
elt = [MarkerElement(color = colm[i], marker = '●', markersize = 10) for i in 1:nlev]
#elt = [PolyElement(polycolor = colm[i]) for i in 1:nlev]
title = "Group"
Legend(f[1, 2], elt, lev, title; nbanks = 1, rowgap = 10, framevisible = false)
f
```
""" 
function umap(X; kwargs...)
    par = recovkw(ParUmap, kwargs).par
    X = ensure_mat(X)
    Q = eltype(X)
    p = nco(X)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X)
        X = fscale(X, xscales)
    end
    fm = UMAP.UMAP_(X', par.nlv; 
        n_neighbors = par.n_neighbors, 
        metric = Distances.Euclidean(), 
        min_dist = par.min_dist
        )
    T = fm.embedding' 
    Umap(T, fm, xscales, par)
end

""" 
    transf(object::Umap, X)
Compute latent variables (LVs = scores T) from a fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
"""
function transf(object::Umap, X)
    X = ensure_mat(X)
    UMAP.transform(object.fm, fscale(X, object.xscales)')'
end

