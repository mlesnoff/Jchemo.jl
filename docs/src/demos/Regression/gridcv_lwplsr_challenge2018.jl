mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "challenge2018.jld2") 
@load db dat
pnames(dat)

X = dat.X 
Y = dat.Y
wl = names(X)
wl_num = parse.(Float64, wl)
ntot = nro(X)

# Preprocessing
f = 21 ; pol = 3 ; d = 2 ;
Xp = savgol(snv(X); f = f, pol = pol, d = d) ;
plotsp(Xp; nsamp = 20).f

y = Y.conc

s = Bool.(Y.test)
Xtrain = rmrow(Xp, s)
ytrain = rmrow(y, s)
Xtest = Xp[s, :]
ytest = y[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
(ntot = ntot, ntrain, ntest)

######## End Data

m = 300 ; segm = segmts(ntrain, m; rep = 3)     # Test-set CV
#K = 3 ; segm = segmkf(ntrain, K; rep = 1)      # K-fold CV      

## PLSR
nlv = 0:50
res = gridcvlv(Xtrain, ytrain; segm = segm, 
    score = rmsep, fun = plskern, nlv = nlv, verbose = true).res
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]

plotgrid(res.nlv, res.y1; step = 10,
    xlabel = "Nb. LVs", ylabel = "RMSEP").f

fm = plskern(Xtrain, ytrain; nlv = res.nlv[u]) ;
pred = Jchemo.predict(fm, Xtest).pred
rmsep(pred, ytest)

plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f

## LWPLSR
nlvdis = [15; 25] ; metric = ["mahal"] 
h = [1; 2.5; 5; Inf] ; k = [150; 250; 500; 1000]  
nlv = 1:25 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k)
length(pars[1])
nlv = 0:25
res = gridcvlv(Xtrain, ytrain; segm = segm, score = rmsep, 
    fun = lwplsr, nlv = nlv, pars = pars, verbose = true).res
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]

group = string.(res.metric, res.nlvdis, " h=", res.h, " k=", res.k)
plotgrid(res.nlv, res.y1, group; step = 2,
    xlabel = "Nb. LVs", ylabel = "RMSEP").f

fm = lwplsr(Xtrain, ytrain; nlvdis = res.nlvdis[u], 
    metric = res.metric[u], h = res.h[u], k = res.k[u], 
    nlv = res.nlv[u], verbose = true) ;
pred = Jchemo.predict(fm, Xtest).pred
rmsep(pred, ytest)

plotxy(vec(pred), ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", ylabel = "Observed").f

## Averaging

nlvdis = [15; 25] ; metric = ["mahal"] 
h = [1; 2.5] ; k = [150; 250; 300; 500]  
nlv = ["0:20"; "0:30"; "0:50"; "5:20"; "5:30"; "5:50"] 
pars = mpar(nlv = nlv, nlvdis = nlvdis, metric = metric, h = h, k = k)
length(pars[1])
res = gridcv(Xtrain, ytrain; segm = segm, score = rmsep, 
    fun = lwplsr_avg, pars = pars, verbose = true).res
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]

fm = lwplsr_avg(Xtrain, ytrain; nlvdis = res.nlvdis[u], 
    metric = res.metric[u], h = res.h[u], k = res.k[u], 
    nlv = res.nlv[u], verbose = true) ;
pred = Jchemo.predict(fm, Xtest).pred
rmsep(pred, ytest)

## Averaging - direct
nlv = "5:20"
#nlv = "0:30"
#nlv = "0:50"
fm = lwplsr_avg(Xtrain, ytrain; nlvdis = 15, 
    metric = "mahal", h = 2, k = 200, 
    nlv = nlv, verbose = true) ;
pred = Jchemo.predict(fm, Xtest).pred
rmsep(pred, ytest)

## Random forests - direct
fm = rfr_xgb(Xtrain, ytrain; rep = 50,
    subsample = .7, colsample_bynode = 1 / 3,
    max_depth = 2000, min_child_weight = 5) ;
pred = Jchemo.predict(fm, Xtest).pred ;
rmsep(pred, ytest)

## XGBoost - direct
fm = xgboostr(Xtrain, ytrain; rep = 150,
    eta = .1, subsample = .7,
    colsample_bytree = 1/3, colsample_bynode = 1/3,
    max_depth = 6, min_child_weight = 5,
    lambda = .3) ;
pred = Jchemo.predict(fm, Xtest).pred ;
rmsep(pred, ytest)


