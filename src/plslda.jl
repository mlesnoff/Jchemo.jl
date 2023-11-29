"""
    plslda(X, y, weights = ones(nro(X)); nlv, 
        prior = :unif, scal::Bool = false)
PLS-LDA.
* `X` : X-data.
* `y` : y-data (class membership).
* `weights` : Weights of the observations. 
    Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `prior` : Type of prior probabilities for class membership.
    Possible values are: :unif (uniform; default), :prop (proportional).
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

LDA on PLS latent variables.

The training variable `y` (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy variable (0/1). 
Then, a PLS2 is implemented on `X` and Ydummy, 
returning `nlv` latent variables (LVs). Finally, a LDA is run on these LVs and `y`. 

## Examples
```julia
using JLD2
using JchemoData
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2") 
@load db dat
pnames(dat)

X = dat.X 
Y = dat.Y 
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]

tab(ytrain)
tab(ytest)

## nlv must be >=1 
## (conversely to plsrda for which nlv >= 0)
nlv = 20      
fm = plslda(Xtrain, ytrain; nlv = nlv) ;    
#fm = plsqda(Xtrain, ytrain; nlv = nlv) ;
pnames(fm)
pnames(fm.fm)

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.pred
err(res.pred, ytest)
confusion(res.pred, ytest).cnt

transf(fm, Xtest)
transf(fm, Xtest; nlv = 2)

fmpls = fm.fm.fmpls ;
transf(fmpls, Xtest)
summary(fmpls, Xtrain)
Jchemo.coef(fmpls).B
Jchemo.coef(fmpls, nlv = 1).B
Jchemo.coef(fmpls, nlv = 2).B

fmda = fm.fm.fmda ;
T = transf(fmpls, Xtest)
Jchemo.predict(fmda[nlv], T).pred

Jchemo.predict(fm, Xtest; nlv = 1:2).pred
```
""" 
function plslda(X, y, weights = ones(nro(X)); nlv, 
        prior = :unif, scal::Bool = false)
    res = dummy(y)
    ni = tab(y).vals
    fmpls = plskern(X, res.Y, weights; nlv = nlv, scal = scal)
    fmda = list(nlv)
    @inbounds for i = 1:nlv
        fmda[i] = lda(fmpls.T[:, 1:i], y, weights; prior = prior)
    end
    fm = (fmpls = fmpls, fmda = fmda)
    Plslda(fm, res.lev, ni)
end

""" 
    transf(object::Plslda, X; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and a matrix X.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Plslda, X; nlv = nothing)
    transf(object.fm.fmpls, X; nlv = nlv)
end

"""
    predict(object::Plslda, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
   
""" 
function predict(object::Plslda, X; nlv = nothing)
    X = ensure_mat(X)
    m = nro(X)
    a = size(object.fm.fmpls.T, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(le_nlv, Union{Matrix{Int}, Matrix{Float64}, Matrix{String}})
    posterior = list(le_nlv, Matrix{Float64})
    @inbounds for i = 1:le_nlv
        znlv = nlv[i]
        T = transf(object.fm.fmpls, X, nlv = znlv)
        zres = predict(object.fm.fmda[znlv], T)
        z =  mapslices(argmax, zres.posterior; dims = 2) 
        pred[i] = reshape(replacebylev2(z, object.lev), m, 1)
        posterior[i] = zres.posterior
    end 
    if le_nlv == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end





