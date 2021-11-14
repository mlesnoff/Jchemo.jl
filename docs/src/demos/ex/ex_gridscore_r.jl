using Jchemo

n = 50 ; p = 7 ; q = 2 
Xtrain = rand(n, p) ; Ytrain = rand(n, q) 
ytrain = Ytrain[:, 1] 
m = 3 
Xtest = rand(m, p) ; Ytest = rand(m, q) 
ytest = Ytest[:, 1] 

########### PLSR

nlv = 0:2 
pars = mpars(nlv = nlv)
gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = plskern, pars = pars)

# Faster
gridscorelv(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = plskern, nlv = nlv)

########### KPLSR

nlv = 0:2 
pars = mpars(nlv = nlv, gamma = [1; 10])
#pars = mpars(nlv = 0:nlv, kern = ["kpol";], degree = 1:2)
gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = kplsr, pars = pars)

# Faster
pars = mpars(gamma = [1, 10])
gridscorelv(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = kplsr, nlv = nlv, pars = pars)

########### DKPLSR

nlv = 0:2 
pars = mpars(nlv = nlv, gamma = [1; 10])
gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = dkplsr, pars = pars)

# Faster
pars = mpars(gamma = [1; 10])
gridscorelv(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = dkplsr, nlv = nlv, pars = pars)

############ RR

lb = [.01; .1]
pars = mpars(lb = lb)
gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = rr, pars = pars)

gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = rrchol, pars = pars)

# Faster (only for rr, not for rrchol)
gridscorelb(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = rr, lb = lb)
    
############ KRR

lb = [.01; .1]
pars = mpars(lb = lb, gamma = [1; 10])
gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = krr, pars = pars)

# Faster
pars = mpars(gamma = [1; 10])
gridscorelb(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = krr, lb = lb, pars = pars)

############ kNN-R

nlvdis = 5 ; metric = ["mahal"; "eucl"] 
h = [1; 3] ; k = [20; 10] 
pars = mpars(nlvdis = nlvdis, metric = metric, h = h, k = k) 
gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = knnr, pars = pars, verbose = true)
    
############ kNN-LWPLSR

nlvdis = 5 ; metric = ["mahal";] 
h = [1.; 3.] ; k = [100; 20]
nlv = 1:2 
pars = mpars(nlv = nlv, nlvdis = nlvdis, metric = metric, h = h, k = k) 
gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = lwplsr, pars = pars, verbose = true)

# Faster
pars = mpars(nlvdis = nlvdis, metric = metric, h = h, k = k) 
gridscorelv(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = lwplsr, nlv = nlv, pars = pars, verbose = false)

########### PLSR-AGG

# Here there is no sense to use gridscorelv

pars = mpars(nlv = ["1:2"; "1:3"])
gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = plsr_agg, pars = pars)

############ kNN-LWPLSR-AGG

# Here there is no sense to use gridscorelv

nlvdis = 5 ; metric = ["mahal";] ;
h = [1.; 3.] ; k = [20; 10]
nlv = ["1:2"; "2:5"] ;
pars = mpars(nlv = nlv, nlvdis = nlvdis, metric = metric, h = h, k = k) ;
gridscore(Xtrain, Ytrain, Xtest, Ytest;
    score = msep, fun = lwplsr_agg, pars = pars, verbose = true)




