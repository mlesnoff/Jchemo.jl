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

K = 3 ; segm = segmkf(ntrain, K; rep = 10)       

nlv = 0:30
res = gridcvlv(Xtrain, ytrain; segm = segm, 
    score = rmsep, fun = plskern, nlv = nlv, verbose = true).res
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]

plotgrid(res.nlv, res.y1; step = 2,
    xlabel = "Nb. LVs", ylabel = "RMSEP").f

zres = selwold(res.nlv, res.y1; smooth = true, 
    graph = true) ;
pnames(res)
zres.opt     # Nb. LVs correponding to the minimal error rate
zres.sel     # Nb LVs selected with the Wold's criterion
zres.f       # Plots

# opt
fm = plskern(Xtrain, ytrain; nlv = res.nlv[u]) ;
pred = Jchemo.predict(fm, Xtest).pred
rmsep(pred, ytest)

# parcimonious
fm = plskern(Xtrain, ytrain; nlv = zres.sel) ;
pred = Jchemo.predict(fm, Xtest).pred
rmsep(pred, ytest)


