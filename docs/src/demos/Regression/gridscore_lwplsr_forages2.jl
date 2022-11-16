mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "forages2.jld2") 
@load db dat
pnames(dat)
  
X = dat.X 
Y = dat.Y
wl = names(X)
wl_num = parse.(Float64, wl)
ntot = nro(X)

plotsp(X, wl_num; nsamp = 5).f
summ(Y)

y = Y.ndf
#y = Y.dm

s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(y, s)
Xtest = X[s, :]
ytest = y[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
(ntot = ntot, ntrain, ntest)

######## End Data

## Train ==> Cal + Val
pct = .30
nval = Int64.(round(pct * ntrain))
ncal = ntrain - nval 
s = sample(1:ntrain, nval; replace = false)
Xcal = rmrow(Xtrain, s) 
ycal = rmrow(ytrain, s) 
Xval = Xtrain[s, :] 
yval = ytrain[s] 
(ntot = ntot, ntrain, ncal, nval, ntest)
## End

nlvdis = [25] ; metric = ["mahal"]
h = [1; 2; 5] ; k = [100; 250; 500]
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k)
length(pars[1])
nlv = 0:25
res = gridscorelv(Xcal, ycal, Xval, yval; score = rmsep, 
    fun = lwplsr, nlv = nlv, pars = pars, 
    verbose = true)
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

