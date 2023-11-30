"""
    gridscore(Xtrain, Ytrain, X, Y; score, fun, pars, verbose = FALSE) 
Model validation over a grid of parameters.
* `Xtrain` : Training X-data (n, p).
* `Ytrain` : Training Y-data (n, q).
* `X` : Validation X-data (m, p).
* `Y` : Validation Y-data (m, q).
* `score` : Function (e.g. `msep`) computing the prediction score.
* `fun` : Function computing the prediction model.
* `pars` : tuple of named vectors (= arguments of fun) of same length
    involved in the calculation of the score (e.g. output of function `mpar`).
* `verbose` : If true, fitting information are printed.

Compute a prediction score (= error rate) for a given model over a grid of parameters.

The score is computed over the validation sets `X` and `Y` for each combination 
of the grid defined in `pars`. 
    
The vectors in `pars` must have same length.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)

# Building Train (years <= 2012) and Test  (year == 2012)

X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)
ntrain = nro(Xtrain)

# Building Cal and Val within Train

nval = 80
s = sample(1:ntrain, nval; replace = false)
Xcal = rmrow(Xtrain, s)
ycal = rmrow(ytrain, s)
Xval = Xtrain[s, :]
yval = ytrain[s]

# KNNR models

nlvdis = 15 ; metric = [:mah ]
h = [1 ; 2.5] ; k = [5 ; 10 ; 20 ; 50] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k) 
length(pars[1]) 
res = gridscore(Xcal, ycal, Xval, yval;
    score = rmsep, fun = knnr, pars = pars, verbose = true)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]

fm = knnr(Xtrain, ytrain;
    nlvdis = res.nlvdis[u], metric = res.metric[u],
    h = res.h[u], k = res.k[u]) ;
pred = Jchemo.predict(fm, Xtest).pred 
rmsep(pred, ytest)

################# PLSR models

nlv = 0:20
res = gridscorelv(Xcal, ycal, Xval, yval;
    score = rmsep, fun = plskern, nlv = nlv)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
plotgrid(res.nlv, res.y1;
    xlabel = "Nb. LVs", ylabel = "RMSEP").f

fm = plskern(Xtrain, ytrain; nlv = res.nlv[u]) ;
pred = Jchemo.predict(fm, Xtest).pred 
rmsep(pred, ytest)

# LWPLSR models

nlvdis = 15 ; metric = [:mah ]
h = [1 ; 2.5 ; 5] ; k = [50 ; 100] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k)
length(pars[1]) 
nlv = 0:20
res = gridscorelv(Xcal, ycal, Xval, yval;
    score = rmsep, fun = lwplsr, pars = pars, nlv = nlv, verbose = true)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
group = string.("h=", res.h, " k=", res.k)
plotgrid(res.nlv, res.y1, group;
    xlabel = "Nb. LVs", ylabel = "RMSECV").f

fm = lwplsr(Xtrain, ytrain;
    nlvdis = res.nlvdis[u], metric = res.metric[u],
    h = res.h[u], k = res.k[u], nlv = res.nlv[u]) ;
pred = Jchemo.predict(fm, Xtest).pred 
rmsep(pred, ytest)

################# RR models

lb = (10.).^collect(-5:1:-1)
res = gridscorelb(Xcal, ycal, Xval, yval;
    score = rmsep, fun = rr, lb = lb)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
plotgrid(log.(res.lb), res.y1;
    xlabel = "Lambda", ylabel = "RMSECV").f

fm = rr(Xtrain, ytrain; lb = res.lb[u]) ;
pred = Jchemo.predict(fm, Xtest).pred 
rmsep(pred, ytest)

################# KRR models

gamma = (10.).^collect(-4:1:4)
pars = mpar(gamma = gamma)
length(pars[1]) 
lb = (10.).^collect(-5:1:-1)
res = gridscorelb(Xcal, ycal, Xval, yval;
    score = rmsep, fun = krr, pars = pars, lb = lb)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
group = string.("gamma=", res.gamma)
plotgrid(log.(res.lb), res.y1, group;
    xlabel = "Lambda", ylabel = "RMSECV").f

fm = krr(Xtrain, ytrain; gamma = res.gamma[u], lb = res.lb[u]) ;
pred = Jchemo.predict(fm, Xtest).pred 
rmsep(pred, ytest)
```
"""
function gridscore(Xtrain, Ytrain, X, Y; fun, score, 
        pars = nothing, nlv = nothing, lb = nothing, 
        verbose = false)
    if isnothing(nlv) && isnothing(lb)
        res = gridscorebr(Xtrain, Ytrain, X, Y; fun, score, 
            pars, verbose = verbose)
    elseif !isnothing(nlv)
        res = gridscorelv(Xtrain, Ytrain, X, Y; fun, score, 
            pars, nlv, verbose)
    elseif !isnothing(lb)
        res = gridscorelb(Xtrain, Ytrain, X, Y; fun, score, 
            pars, lb, verbose)
    end
    res
end

