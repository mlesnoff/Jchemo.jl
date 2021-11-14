using Jchemo, LinearAlgebra

n = 50 ; p = 7 
Xtrain = rand(n, p) 
m = 3 
Xtest = rand(m, p)  

nlv = 5
fm = pcasvd(Xtrain; nlv = nlv) ;
#fm = pcaeigen(Xtrain; nlv = nlv) ;
pnames(fm)
fm.T
fm.T' * fm.T
fm.P' * fm.P

Jchemo.transform(fm, Xtest)

res = Jchemo.summary(fm, Xtrain) ;
pnames(res)
res.explvar
res.contr_var
res.coord_var
res.cor_circle

# Weighted PCA
w = collect(1:n) 
nlv = 5
fm = pcasvd(Xtrain, w; nlv = nlv) ;
D = Diagonal(mweights(w)) ;
fm.T' * D * fm.T
fm.P' * fm.P

