"""
    loess(X, y; kwargs...)
Compute a locally weighted regression model (LOESS).
* `X` : X-data (n, p).
* `y` : Univariate y-data (n).
Keyword arguments:
* `span` : Window for neighborhood selection (level of smoothing) smoothing, 
    typically in [0, 1] (proportion).
* `degree` : Polynomial degree for the local fitting.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
    
The function fits a LOESS model using package `Loess.jl'. 

Smaller values of `span` result in smaller local context in fitting (less smoothing).

## References

https://github.com/JuliaStats/Loess.jl

Cleveland, W. S. (1979). Robust locally weighted regression and smoothing 
scatterplots. Journal of the American statistical association, 74(368), 829-836. 
DOI: 10.1080/01621459.1979.10481038

Cleveland, W. S., & Devlin, S. J. (1988). Locally weighted regression: an approach 
to regression analysis by local fitting. Journal of the American statistical association, 
83(403), 596-610. DOI: 10.1080/01621459.1988.10478639

Cleveland, W. S., & Grosse, E. (1991). Computational methods for local regression. 
Statistics and computing, 1(1), 47-62. DOI: 10.1007/BF01890836

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
mod = model(loess; nlv, n_neighbors, min_dist)  
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
function loess(X, y; kwargs...)
    par = recovkw(ParLoess, kwargs).par
    X = ensure_mat(X)
    Q = eltype(X)
    y = vec(y)
    p = nco(X)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X)
        X = fscale(X, xscales)
    end
    fm = loess(X, y; span = par.span, degree = par.degree) 
    Loess(fm, xscales, par) 
end

"""
    predict(object::Loess, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Loess, X)
    X = ensure_mat(X)
    Q = eltype(X)
    pred = Loess.predict(object.fm, fscale(X, object.xscales))
    #m = length(pred)
    #pred = reshape(convert.(Q, pred), m, 1)
    (pred = pred,)
end

