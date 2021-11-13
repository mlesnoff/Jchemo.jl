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
transform(fm, Xtrain)
transform(fm, Xtrain; nlv = 1)
transform(fm, Xtest)

coef(fm).B
coef(fm).int
coef(fm; nlv = 2).B
coef(fm; nlv = 2).int
coef(fm; nlv = 0).B
coef(fm; nlv = 0).int

predict(fm, Xtest).pred
predict(fm, Xtest; nlv = nlv).pred

predict(fm, Xtest; nlv = 0:3).pred 
predict(fm, Xtest; nlv = 0).pred

pred = predict(fm, Xtest).pred ;
msep(pred, Ytest)

gridscorelv(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = plskern, nlv = 0:nlv)

### Weighted PLS

w = collect(1:n) 
fm = plskern(Xtrain, Ytrain, w; nlv = nlv) ;

