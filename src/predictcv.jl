"""
    predictcv(model, X, Y; segm, score)
Return the data and predictions from a cross-validated model.
* `model` : Model to evaluate.
* `X` : Training X-data (n, p).
* `Y` : Training Y-data (n, q).
Keyword arguments: 
* `segm` : Segments of observations used for the CV (output of functions [`segmts`](@ref), [`segmkf`](@ref), etc.).
* `score` : Function computing the prediction score (e.g. `rmsep`).

## Examples
```julia
using JLD2, CairoMakie, JchemoData
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
@names dat
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
model = savgol(npoint = 21, deriv = 2, degree = 2)
fit!(model, X)
Xp = transf(model, X)
s = year .<= 2012
Xtrain = Xp[s, :]
ytrain = y[s]
Xtest = rmrow(Xp, s)
ytest = rmrow(y, s)
ntrain = nro(Xtrain)
ntest = nro(Xtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)

K = 3 ; rep = 10
segm = segmkf(ntrain, K; rep)
model = plskern()
pars = mpar(scal = [true])
rescv = gridcv(model, Xtrain, ytrain; segm, score = rmsep, pars, nlv = 0:10)
res = rescv.res
plotgrid(res.nlv, res.y1; step = 2, xlabel = "Nb. LVs", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]  # best model

## Within-CV predictions for the best model
model = plskern(nlv = res.nlv[u], scal = res.scal[u])
respred = predictcv(model, Xtrain, ytrain; segm, score = rmsep) ; 
@names respred
matpred = respred.matpred
maty = respred.maty
respred.scor  # score computed as in gridcv 

plotxy(matpred.y1, maty.y1; bisect = true,  color = (:red, .3), xlabel = "Prediction", 
    ylabel = "Observed").f   
```
"""
predictcv = function(model, X, Y; segm, score)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    q = nco(Y)
    Q = eltype(Y)
    rep = length(segm)
    K = length(segm[1]) 
    matpred = Vector{Matrix{Q}}(undef, rep * K)
    maty = Vector{Matrix{Q}}(undef, rep * K)
    k = 1
    scor = repeat([0], 1, q)
    for i = 1:rep
        listsegm = segm[i]
        for j = 1:K
            s = listsegm[j]
            m = length(s)
            dat = hcat(repeat([i], m), repeat([j], m))           
            fit!(model, rmrow(X, s), rmrow(Y, s))
            pred = predict(model, vrow(X, s)).pred
            matpred[k] = hcat(dat, pred)
            maty[k] = hcat(dat, vrow(Y, s))
            scor = scor + score(pred, vrow(Y, s))
            k = k + 1
        end
    end
    nam = vcat([:rep, :segm], Symbol.(string.("y", 1:q)))
    typ = vcat([Int; Int], repeat([Q], q))
    mat = DataFrame(reduce(vcat, matpred), nam)
    matpred = convertdf(mat; typ)   
    mat = DataFrame(reduce(vcat, maty), nam)
    maty = convertdf(mat; typ)   
    scor = scor / (rep * K)
    (matpred = matpred, maty, scor)
end



