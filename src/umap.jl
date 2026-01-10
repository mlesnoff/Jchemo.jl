"""
    umap(; kwargs...)
    umap(X; kwargs...)
UMAP: Uniform manifold approximation and projection for dimension reduction
* `X` : X-data (n, p).
Keyword arguments:
* `psamp` : Proportion of sampling in `X` for training.
* `nlv` : Nb. latent variables (LVs) to compute.
* `metric` : Distance metric used. This can be any subtype of the `SemiMetric` type from 
    the `Distances.jl` package, including user-defined types. Default is `Distances.Euclidean()`.
* `n_neighbors` : Nb. approximate neighbors used to construct the initial high-dimensional graph.
* `min_dist` : Minimum distance between points in low-dimensional space.
* `scal` : Boolean. If `true`, each column of `X` and `Y` is scaled by its uncorrected standard deviation.
    
The function fits a UMAP dimension reduction using package `UMAP.jl'.

If `psamp < 1`, only a proportion `psamp` of the observations (rows of `X`) are used to build the model (systematic 
sampling over the first score of the PCA of `X`). Can be used to decrease computation times when n is large.

## References

https://github.com/dillondaudert/UMAP.jl

McInnes, L, Healy, J, Melville, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. 
ArXiV 1802.03426, 2018. https://arxiv.org/abs/1802.03426

https://umap-learn.readthedocs.io/en/latest/how_umap_works.html

https://pair-code.github.io/understanding-umap/ 

## Examples
```julia
using Jchemo, JLD2, DataFrames, GLMakie
using Distances

using JchemoData
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "challenge2018.jld2") 
@load db dat
@names dat
X = dat.X 
Y = dat.Y
wlst = names(X)
wl = parse.(Float64, wlst)
ntot = nro(X)
summ(Y)
y = Y.conc
ycla = Y.typ
test = Y.test
## Preprocessing
model1 = snv() 
model2 = savgol(npoint = 21, deriv = 2, degree = 3)
model = pip(model1, model2)
fit!(model, X)
@head Xp = transf(model, X)
plotsp(Xp, wl; xlabel = "Wavelength (nm)", ylabel = "Absorbance", nsamp = 20).f
## Tot => Train + Test
s = Bool.(test)
Xtrain = rmrow(Xp, s)
ytrain = rmrow(y, s)
yclatrain = rmrow(ycla, s)
Xtest = Xp[s, :]
ytest = y[s]
yclatest = ycla[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
(ntot = ntot, ntrain, ntest)
tab(string.(ycla, "-", Y.label))
##### End

psamp = .2  # to decrease the computation time for the example
#psamp = 1  # all samples
nlv = 3
metric = Distances.Euclidean()
#metric = Distances.CosineDist()
#metric = Jchemo.SamDist()
n_neighbors = 20 ; min_dist = .4 
model = umap(; psamp, nlv, metric, n_neighbors, min_dist)  
fit!(model, Xtrain)
fitm = model.fitm ;
@names fitm 
@head T = fitm.T
@head T_all = transf(model, Xtrain)   # full training scores after refitting 
@head Ttest = transf(model, Xtest)

s = fitm.s
zycla = yclatrain[s]
lev = mlev(zycla)
nlev = length(lev)
colm = cgrad(:tab10, nlev; categorical = true, alpha = .5)
i = 1
f, ax = plotxyz(T[:, i], T[:, i + 1], T[:, i + 2], zycla; size = (700, 500), color = colm, markersize = 10, 
    title = "Umap score space", xlabel = string("LV", i), ylabel = string("LV", i + 1), zlabel = string("LV", i + 2))
scatter!(ax, Ttest[:, i], Ttest[:, i + 1], Ttest[:, i + 2], color = :black, colormap = :tab20, markersize = 7)
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
        ns = round(Int, par.psamp * n)
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
    ## Note: UMAP.jl ==> the type of new_data must match the original data exactly ==> force to Matrix
    fitm = UMAP.fit(Matrix(X'), par.nlv; metric = par.metric, n_neighbors = par.n_neighbors, min_dist = par.min_dist)
    T = reduce(vcat, transpose.(fitm.embedding))
    Umap(fitm, T, xscales, s, par)
end

""" 
    transf(object::Umap, X)
Compute latent variables (LVs; = scores) from a fitted model.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
"""
function transf(object::Umap, X)
    res = UMAP.transform(object.fitm, Matrix(fscale(X, object.xscales)'))
    reduce(vcat, transpose.(res.embedding)) 
end

