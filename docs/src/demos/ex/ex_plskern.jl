n = 6 ; p = 7 ; q = 2 ; m = 3 ;
Xtrain = rand(n, p) ; Ytrain = rand(n, q) ;
Xtest = rand(m, p) ; Ytest = rand(m, q) ;

nlv = 5 ; 
fm = plskern(Xtrain, Ytrain; nlv = nlv)
propertynames(fm)
fm.T
fm.P 

summary(fm, Xtrain).explvar

fm.T
transform(fm, Xtrain)
transform(fm, Xtrain; nlv = 1)
transform(fm, Xtest)

coef(fm).int
coef(fm).B
coef(fm; nlv = 2).int
coef(fm; nlv = 2).B
coef(fm; nlv = 0).int
coef(fm; nlv = 0).B

predict(fm, Xtest).pred
predict(fm, Xtest; nlv = nlv).pred

predict(fm, Xtest; nlv = 0:3).pred 
predict(fm, Xtest; nlv = 0).pred

pred = predict(fm, Xtest).pred ;
msep(pred, Ytest)

gridscorelv(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = plskern, nlv = 0:nlv)

### Weighted PLS

weights = collect(1:n) ;
fm = plskern(Xtrain, Ytrain, weights; nlv = nlv)

