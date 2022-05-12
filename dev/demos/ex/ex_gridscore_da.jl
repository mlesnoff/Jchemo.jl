using Jchemo

n = 100 ; p = 10 
X = rand(n, p) 
y = rand([3, 4, 10], n) 
#y = rand(["a", "b", "c"], n) ;
Xtrain = X[1:70, :] ; ytrain = y[1:70] 
Xtest = X[71:n, :] ; ytest = y[71:n]

tab(ytrain)
tab(ytest)

############ PLSR-DA

nlv = 0:3 
pars = mpar(nlv = nlv)
gridscore(Xtrain, ytrain, Xtest, ytest; 
    score = err, fun = plsrda, pars = pars)

# Faster
gridscorelv(Xtrain, ytrain, Xtest, ytest; 
    score = err, fun = plsrda, nlv = nlv)

############ PLS-LDA/QDA

nlv = 1:3 
pars = mpar(nlv = nlv)
gridscore(Xtrain, ytrain, Xtest, ytest; 
    score = err, fun = plslda, pars = pars)

# Faster
gridscorelv(Xtrain, ytrain, Xtest, ytest; 
    score = err, fun = plslda, nlv = nlv)

gridscorelv(Xtrain, ytrain, Xtest, ytest; 
    score = err, fun = plsqda, nlv = nlv)
    
############ RR-DA

lb = [.001; .01; .1] 
pars = mpar(lb = lb)
gridscore(Xtrain, ytrain, Xtest, ytest; 
    score = err, fun = rrda, pars = pars)

# Faster
gridscorelb(Xtrain, ytrain, Xtest, ytest; 
    score = err, fun = rrda, lb = lb)

############ KRR-DA

lb = [.001; .01; .1] 
pars = mpar(lb = lb, gamma = [1; 10])
#pars = mpar(lb = lb, kern = ["kpol";], degree = 1:2)
gridscore(Xtrain, ytrain, Xtest, ytest; 
    score = err, fun = krrda, pars = pars)

# Faster
pars = mpar(gamma = [1; 10])
gridscorelb(Xtrain, ytrain, Xtest, ytest; 
    score = err, fun = krrda, lb = lb, pars = pars)

############ kNN-DA

nlvdis = 5 ; metric = ["mahal";] 
h = [1; 3] ; k = [20; 10] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k) 
gridscore(Xtrain, ytrain, Xtest, ytest;
    score = msep, fun = knnda, pars = pars, verbose = true)
    
############ kNN-LWPLSR-DA

nlvdis = 5 ; metric = ["mahal";] 
h = [1; 3] ; k = [20; 10]
nlv = 1:2 
pars = mpar(nlv = nlv, nlvdis = nlvdis, metric = metric, h = h, k = k) 
gridscore(Xtrain, ytrain, Xtest, ytest;
    score = msep, fun = lwplsrda, pars = pars, verbose = true)

# Faster
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k) 
gridscorelv(Xtrain, ytrain, Xtest, ytest;
    score = msep, fun = lwplsrda, nlv = nlv, pars = pars, verbose = false)

############ kNN-LWPLS-LDA

nlvdis = 5 ; metric = ["mahal";] 
h = [1; 3] ; k = [20; 10]
nlv = 1:2 
pars = mpar(nlv = nlv, nlvdis = nlvdis, metric = metric, h = h, k = k) 
gridscore(Xtrain, ytrain, Xtest, ytest;
    score = msep, fun = lwplslda, pars = pars, verbose = true)

# Faster
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k) 
gridscorelv(Xtrain, ytrain, Xtest, ytest;
    score = msep, fun = lwplslda, nlv = nlv, pars = pars, verbose = false)

########### PLSR-DA-AGG

# Here there is no sense to use gridscorelv

pars = mpar(nlv = ["1:2"; "1:3"])
gridscore(Xtrain, ytrain, Xtest, ytest;
    score = msep, fun = plsrda_avg, pars = pars)

############ kNN-LWPLSR-DA-AGG

# Here there is no sense to use gridscorelv

nlvdis = 5 ; metric = ["mahal";] ;
h = [1.; 3.] ; k = [20; 10]
nlv = ["1:2"; "2:5"] ;
pars = mpar(nlv = nlv, nlvdis = nlvdis, metric = metric, h = h, k = k) ;
gridscore(Xtrain, ytrain, Xtest, ytest;
    score = msep, fun = lwplsrda_avg, pars = pars, verbose = true)



