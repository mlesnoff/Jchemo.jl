"""
    dkplsrda(X, y, weights = ones(nro(X)); nlv, kern = :krbf, 
        scal::Bool = false, kwargs...)
Discrimination based on direct kernel partial least squares regression (DKPLSR-DA).
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations. Internally normalized to sum to 1. 
* `nlv` : Nb. latent variables (LVs) to compute.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
* Other arguments to pass in the kernel: See `?kplsr`.

This is the same approach as for `plsrda` except that the PLS2 step 
is replaced by a non linear direct kernel PLS2 (DKPLS).

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

gamma = .001 
nlv = 15
fm = dkplsrda(Xtrain, ytrain; nlv = nlv, gamma = gamma) ;
pnames(fm)
typeof(fm.fm) # = KPLS2 model

res = Jchemo.predict(fm, Xtest) ;
pnames(res)
res.posterior
res.pred
err(res.pred, ytest)
confusion(res.pred, ytest).cnt

transf(fm, Xtest; nlv = 2)

transf(fm.fm, Xtest)
Jchemo.coef(fm.fm)
```
""" 
function dkplsrda(X, y, weights = ones(nro(X)); nlv, kern = :krbf, 
        scal::Bool = false, kwargs...)
    res = dummy(y)
    ni = tab(y).vals
    fm = dkplsr(X, res.Y, weights; nlv = nlv, kern = kern, 
        scal = scal, kwargs...)
    Dkplsrda(fm, res.lev, ni)
end

""" 
    transf(object::Dkplsrda, X; nlv = nothing)
Compute latent variables (LVs = scores T) from a fitted model and a matrix X.
* `object` : The fitted model.
* `X` : Matrix (m, p) for which LVs are computed.
* `nlv` : Nb. LVs to consider.
""" 
function transf(object::Dkplsrda, X; nlv = nothing)
    transf(object.fm, X; nlv = nlv)
end

"""
    predict(object::Dkplsrda, X; nlv = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `nlv` : Nb. LVs, or collection of nb. LVs, to consider. 
If nothing, it is the maximum nb. LVs.
""" 
function predict(object::Dkplsrda, X; nlv = nothing)
    X = ensure_mat(X)
    m = nro(X)
    a = size(object.fm.fm.T, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(le_nlv, Union{Matrix{Int}, Matrix{Float64}, Matrix{String}})
    posterior = list(le_nlv, Matrix{Float64})
    @inbounds for i = 1:le_nlv
        zp = predict(object.fm, X; nlv = nlv[i]).pred
        #if softmax
        #    @inbounds for j = 1:m
        #        zp[j, :] .= mweight(exp.(zp[j, :]))
        #   end
        #end
        z =  mapslices(argmax, zp; dims = 2)  # if equal, argmax takes the first
        pred[i] = reshape(replacebylev2(z, object.lev), m, 1)     
        posterior[i] = zp
    end 
    if le_nlv == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end