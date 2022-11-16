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

nlv = 0:30
res = gridscorelv(Xcal, ycal, Xval, yval; 
    score = err, fun = plsrda, nlv = nlv)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]

plotgrid(res.nlv, res.y1; step = 2,
    xlabel = "Nb. LVs", ylabel = "ERR").f

fm = plsrda(Xtrain, ytrain; nlv = res.nlv[u]) ;
pred = Jchemo.predict(fm, Xtest).pred
err(pred, ytest)
freqtable(vec(pred), ytest)

