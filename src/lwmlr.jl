struct Lwmlr
    X::Array{Float64}
    Y::Array{Float64}
    metric::String
    h::Real
    k::Int
    tol::Real
    verbose::Bool
end

"""
    lwmlr(X, Y; metric = "eucl", h, k, 
        tol = 1e-4, verbose = false)
k-Nearest-Neighbours locally weighted multiple linear regression (kNN-LWMLR).
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `metric` : Type of dissimilarity used to select the neighbors and compute
    the weights. Possible values are "eucl" (default; Euclidean distance) 
    and "mahal" (Mahalanobis distance).
* `h` : A scalar defining the shape of the weight function. Lower is h, 
    sharper is the function. See function `wdist`.
* `k` : The number of nearest neighbors to select for each observation to predict.
* `tol` : For stabilization when very close neighbors.
* `verbose` : If true, fitting information are printed.

This is the same principle as function `lwplsr` except that MLR models
are fitted (on the neighborhoods) instead of PLSR models.  The neighborhoods 
are computed on `X` (there is no preliminary dimension reduction).

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

nlv = 20
zfm = pcasvd(Xtrain; nlv = nlv) ;
Ttrain = zfm.T 
Ttest = Jchemo.transform(zfm, Xtest)

fm = lwmlr(Ttrain, ytrain; metric = "mahal",
    h = 2, k = 100) ;
pred = Jchemo.predict(fm, Ttest).pred
println(rmsep(pred, ytest))
plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed (Test)").f  

####### Example of fitting the function sinc(x)
####### described in Rosipal & Trejo 2001 J of Machine Learning Res. p. 105-106 
x = collect(-10:.2:10) 
x[x .== 0] .= 1e-5
n = length(x)
zy = sin.(abs.(x)) ./ abs.(x) 
y = zy + .2 * randn(n) 
fm = lwmlr(x, y; metric = "eucl", h = 1, k = 20) ;
pred = Jchemo.predict(fm, x).pred 
f = Figure(resolution = (700, 300))
ax = Axis(f[1, 1])
scatter!(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "Fitted model")
f[1, 2] = Legend(f, ax, framevisible = false)
f
```
""" 
function lwmlr(X, Y; metric = "eucl", 
        h, k, tol = 1e-4, verbose = false)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    Lwmlr(X, Y, metric, h, k, tol, 
        verbose)
end

"""
    predict(object::Lwmlr, X)
Compute the Y-predictions from the fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Lwmlr, X)
    X = ensure_mat(X)
    m = nro(X)
    # Getknn
    res = getknn(object.X, X; 
        k = object.k, metric = object.metric)
    listw = copy(res.d)
    Threads.@threads for i = 1:m
        w = wdist(res.d[i]; h = object.h)
        w[w .< object.tol] .= object.tol
        listw[i] = w
    end
    # End
    pred = locw(object.X, object.Y, X; 
        listnn = res.ind, listw = listw, fun = mlr,
        verbose = object.verbose).pred
    (pred = pred, listnn = res.ind, listd = res.d, 
        listw = listw)
end

