using Jchemo

n = 50 ; p = 7 ; q = 2 
Xtrain = rand(n, p) ; Ytrain = rand(n, q) 
ytrain = Ytrain[:, 1] 
m = 3 
Xtest = rand(m, p) ; Ytest = rand(m, q) 
ytest = Ytest[:, 1] 

########### PLSR

nlv = 0:2 
pars = mpar(nlv = nlv)
res = gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = plskern, pars = pars)

# Faster
res = gridscorelv(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = plskern, nlv = nlv)

########### KPLSR

nlv = 0:2 
pars = mpar(nlv = nlv, gamma = [1; 10])
#pars = mpar(nlv = 0:nlv, kern = ["kpol"], degree = 1:2)
res = gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = kplsr, pars = pars)

# Faster
pars = mpar(gamma = [1, 10])
res = gridscorelv(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = kplsr, nlv = nlv, pars = pars)

########### DKPLSR

nlv = 0:2 
pars = mpar(nlv = nlv, gamma = [1; 10])
res = gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = dkplsr, pars = pars)

# Faster
pars = mpar(gamma = [1; 10])
res = gridscorelv(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = dkplsr, nlv = nlv, pars = pars)

############ RR

lb = [.01; .1]
pars = mpar(lb = lb)
res = gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = rr, pars = pars)

res = gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = rrchol, pars = pars)

# Faster (only for rr, not for rrchol)
res = gridscorelb(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = rr, lb = lb)
    
############ KRR

lb = [.01; .1]
pars = mpar(lb = lb, gamma = [1; 10])
res = gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = krr, pars = pars)

# Faster
pars = mpar(gamma = [1; 10])
res = gridscorelb(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = krr, lb = lb, pars = pars)

############ kNN-R

nlvdis = 5 ; metric = ["mahal"; "eucl"] 
h = [1; 3] ; k = [20; 10] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k) 
res = gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = knnr, pars = pars, verbose = true)
    
############ kNN-LWPLSR

nlvdis = 5 ; metric = ["mahal"] 
h = [1.; 3.] ; k = [100; 20]
nlv = 1:2 
pars = mpar(nlv = nlv, nlvdis = nlvdis, metric = metric, h = h, k = k) 
res = gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = lwplsr, pars = pars, verbose = true)

# Faster
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k) 
res = gridscorelv(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = lwplsr, nlv = nlv, pars = pars, verbose = false)

########### PLSR-AGG

# Here there is no sense to use gridscorelv

pars = mpar(nlv = ["1:2"; "1:3"])
res = gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = plsr_avg, pars = pars)

############ kNN-LWPLSR-AGG

# Here there is no sense to use gridscorelv

nlvdis = 5 ; metric = ["mahal"] ;
h = [1.; 3.] ; k = [20; 10]
nlv = ["1:2"; "2:5"] ;
pars = mpar(nlv = nlv, nlvdis = nlvdis, metric = metric, h = h, k = k) ;
res = gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = rmsep, fun = lwplsr_avg, pars = pars, verbose = true)




