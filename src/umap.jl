"""
    umap(; kwargs...)
    umap(X; kwargs...)
UMAP: Uniform manifold approximation and projection for 
    dimension reduction
* `X` : X-data (n, p).
Keyword arguments:
* `nlv` : Nb. latent variables (LVs) to compute.
* `psamp` : Proportion of sampling in `X` for training.
* `n_neighbors` : Nb. approximate neighbors used to construct 
    the initial high-dimensional graph.
* `min_dist` : Minimum distance between points in low-dimensional 
    space.
* `scal` : Boolean. If `true`, each column of `X` and `Y` 
    is scaled by its uncorrected standard deviation.
    
The function fits a UMAP dimension reducion using 
package `UMAP.jl'. The used metric is the Euclidean distance. 

If `psamp < 1`, only a proportion `psamp` of the observations (rows of `X`) 
are used to build the model (systematic sampling over the first score of 
the PCA of `X`). Can be used to decrease computation times when n
is large.

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

model1 = snv() 
model2 = savgol(npoint = 21, deriv = 2, degree = 3)
model = pip(model1, model2)
fit!(model, X)
@head Xp = transf(model, X)
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
model = umap(; nlv, n_neighbors, min_dist)  
fit!(model, Xtrain)
@head T = model.fitm.T
@head Ttest = transf(model, Xtest)

nlv = 3
n_neighbors = 50 ; min_dist = .5 
model = umap(; nlv, n_neighbors, min_dist)  
fit!(model, Xtrain)
@head T = model.fitm.T
@head Ttest = transf(model, Xtest)
GLMakie.activate!() 
#CairoMakie.activate!()
lev = mlev(typtrain)
nlev = length(lev)
colsh = :tab10
colm = cgrad(colsh, nlev; alpha = .7, categorical = true) 
ztyp = recod_catbyint(typtrain)
f = Figure()
i = 1
ax = Axis3(f[1, 1], xlabel = string("LV", i), ylabel = string("LV", i + 1), 
        zlabel = string("LV", i + 2), title = "UMAP", perspectiveness = 0) 
scatter!(ax, T[:, i], T[:, i + 1], T[:, i + 2]; markersize = 8, 
    color = ztyp, colormap = colm) 
scatter!(ax, Ttest[:, i], Ttest[:, i + 1], Ttest[:, i + 2], color = :black, 
    markersize = 10)  
elt = [MarkerElement(color = colm[i], marker = '●', markersize = 10) for i in 1:nlev]
#elt = [PolyElement(polycolor = colm[i]) for i in 1:nlev]
title = "Group"
Legend(f[1, 2], elt, lev, title; nbanks = 1, rowgap = 10, framevisible = false)
f
```
""" 
umap(; kwargs...) = JchemoModel(umap, nothing, kwargs)

function umap(X; kwargs...)
    par = recovkw(ParUmap, kwargs).par
    X = ensure_mat(X)
    Q = eltype(X)
    n, p = size(X)
    if par.psamp < 1
        ns = Int(round(par.psamp * n))
        res = nipals(fcenter(X, colmean(X)); maxit = 50)
        s = sampsys(res.u, ns).test
        X = vrow(X, s)
    else
        s = collect(1:n)
    end
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X)
        X = fscale(X, xscales)
    end
    metric = Distances.Euclidean()
    fitm = UMAP.UMAP_(X', par.nlv; n_neighbors = par.n_neighbors, metric, min_dist = par.min_dist)
    T = fitm.embedding' 
    Umap(T, fitm, xscales, s, par)
end

""" 
    transf(object::Umap, X)
Compute latent variables (LVs = scores T) from a fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
"""
function transf(object::Umap, X)
    X = ensure_mat(X)
    UMAP.transform(object.fitm, fscale(X, object.xscales)')'
end

