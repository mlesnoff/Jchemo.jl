mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "forages2.jld2") 
@load db dat
pnames(dat)
  
X = dat.X 
Y = dat.Y
y = Y.typ
wl = names(X)
wl_num = parse.(Float64, wl)
ntot = nro(X)

plotsp(X, wl_num).f
summ(Y)
freqtable(y, Y.test)

s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(y, s)
Xtest = X[s, :]
ytest = y[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
(ntot = ntot, ntrain, ntest)

######## End Data

m = 100 ; segm = segmts(ntrain, m; rep = 30)      # Test-set CV
#K = 3 ; segm = segmkf(ntrain, K; rep = 10)       # K-fold CV   

nlv = 0:30
res = gridcvlv(Xtrain, ytrain; segm = segm, 
    score = err, fun = plsrda, nlv = nlv, verbose = true).res
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]

plotgrid(res.nlv, res.y1; step = 2,
    xlabel = "Nb. LVs", ylabel = "ERR").f

fm = plsrda(Xtrain, ytrain; nlv = res.nlv[u]) ;
pred = Jchemo.predict(fm, Xtest).pred
err(pred, ytest)
freqtable(vec(pred), ytest)

## Averaging
nlv = "0:50"
fm = plsrda_avg(Xtrain, ytrain; nlv = nlv) ;
pred = Jchemo.predict(fm, Xtest).pred
err(pred, ytest)
