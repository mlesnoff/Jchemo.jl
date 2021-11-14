using Jchemo

n = 6 ; p = 7 ; q = 2 ; m = 3 
Xtrain = rand(n, p) ; Ytrain = rand(n, q) 
Xtest = rand(m, p) ; Ytest = rand(m, q) 

nlv = 5 ; 
fm = plskern(Xtrain, Ytrain; nlv = nlv) ;
pnames(fm)
fm.T
fm.P 

summary(fm, Xtrain).explvar

fm.T
Jchemo.transform(fm, Xtrain)
Jchemo.transform(fm, Xtrain; nlv = 1)
Jchemo.transform(fm, Xtest)

Jchemo.coef(fm).B
Jchemo.coef(fm).int
Jchemo.coef(fm; nlv = 2).B
Jchemo.coef(fm; nlv = 2).int
Jchemo.coef(fm; nlv = 0).B
Jchemo.coef(fm; nlv = 0).int

Jchemo.predict(fm, Xtest).pred
Jchemo.predict(fm, Xtest; nlv = nlv).pred

Jchemo.predict(fm, Xtest; nlv = 0:3).pred 
Jchemo.predict(fm, Xtest; nlv = 0).pred

pred = Jchemo.predict(fm, Xtest).pred ;
msep(pred, Ytest)

### Weighted PLS

w = collect(1:n) 
fm = plskern(Xtrain, Ytrain, w; nlv = nlv) ;

