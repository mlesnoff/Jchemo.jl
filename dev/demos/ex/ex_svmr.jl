using Jchemo

n = 50 ; p = 7 
Xtrain = rand(n, p) ; ytrain = rand(n) 
m = 3 
Xtest = rand(m, p) ; ytest = rand(m) 

fm = svmr(Xtrain, ytrain) ;
#fm = svmr(Xtrain, ytrain; cost = .1, gamma = 10) ;
#fm = svmr(Xtrain, ytrain; kern = "kpol", degree = 2) ;
res = Jchemo.predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)




