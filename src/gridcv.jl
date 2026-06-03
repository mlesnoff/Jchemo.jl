"""
    gridcv(model, X, Y; segm::Vector{Vector{Vector{Int64}}}, 
        score::Function, pars::Union{Nothing, NamedTuple} = nothing, 
        nlv::Union{Nothing, Int, AbstractVector{Int}} = nothing, 
        lb::Union{Nothing, Float64, AbstractVector{Float64}} = nothing,  
        verbose::Bool = false)
Cross-validation (CV) of a model over a grid of parameters.
* `model` : Model to evaluate.
* `X` : Training X-data (n, p).
* `Y` : Training Y-data (n, q).
Keyword arguments: 
* `segm` : Segments of observations used for the CV (output of functions [`segmts`](@ref), [`segmkf`](@ref), etc.).
* `score` : Function computing the prediction score (e.g., `rmsep`).
* `pars` : tuple of named vectors of same length defining the parameter combinations (e.g., output of function `mpar`).
* `verbose` : If `true`, predicting information are printed.
* `nlv` : Value, or vector of values, of the nb. of latent variables (LVs).
* `lb` : Value, or vector of values, of the ridge regularization parameter "lambda".

The function is used for grid-search: it computes a prediction score (= error rate) for the specified `model` 
for each parameter combination defined in `pars`.
    
For models based on LV or ridge regularization, using arguments `nlv` and `lb` allow faster computations than including 
these parameters in argument `pars. See the examples.   

**For pipeline models:** In the present version of the function, only the last model of the pipeline 
(= the final predictor) is tuned. Therefore, argument `pars` must only contain parameters for this last model.

The function returns two outputs: 
* `res` : mean results
* `res_p` : results per replication.

## Examples
```julia
####### Regression

using Jchemo, JLD2, CairoMakie, JchemoData
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

## a) Replicated K-fold CV 
K = 3 ; rep = 10
segm = segmkf(ntrain, K; rep)
## b) Replicated test-set validation
#m = round(Int, ntrain / 3) ; rep = 30
#segm = segmts(ntrain, m; rep)

####---- Plsr
model = plskern()
nlv = 0:30
rescv = gridcv(model, Xtrain, ytrain; segm, score = rmsep, nlv) ;
@names rescv
res = rescv.res 
plotgrid(res.nlv, res.y1; step = 2, xlabel = "Nb. LVs", ylabel = "RMSEP-CV").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = plskern(; nlv = res.nlv[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

## Example of plot showing replications
res_rep = rescv.res_rep
f, ax = plotgrid(res.nlv, res.y1; step = 2, xlabel = "Nb. LVs", ylabel = "RMSEP-CV")
for i = 1:rep, j = 1:K
    zres = res_rep[res_rep.rep .== i .&& res_rep.segm .== j, :]
    lines!(ax, zres.nlv, zres.y1; color = (:grey, .2))
end
lines!(ax, res.nlv, res.y1; color = :red, linewidth = 1)
f

## Adding pars 
pars = mpar(scal = [false; true])
rescv = gridcv(model, Xtrain, ytrain; segm,  score = rmsep, pars, nlv) ;
res = rescv.res 
typ = res.scal
plotgrid(res.nlv, res.y1, typ; step = 2, xlabel = "Nb. LVs", ylabel = "RMSEP-CV").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = plskern(nlv = res.nlv[u], scal = res.scal[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f   

####---- Rr 
lb = (10).^(-8:.1:3)
model = rr() 
rescv = gridcv(model, Xtrain, ytrain; segm, score = rmsep, lb) ;
res = rescv.res 
loglb = log.(10, res.lb)
plotgrid(loglb, res.y1; step = 2, xlabel = "log(lambda)", ylabel = "RMSEP-CV").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = rr(lb = res.lb[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f     
    
## Adding pars 
pars = mpar(scal = [false; true])
rescv = gridcv(model, Xtrain, ytrain; segm, score = rmsep, pars, lb) ;
res = rescv.res 
loglb = log.(10, res.lb)
typ = string.(res.scal)
plotgrid(loglb, res.y1, typ; step = 2, xlabel = "log(lambda)", ylabel = "RMSEP-CV").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = rr(lb = res.lb[u], scal = res.scal[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f   

####---- Kplsr 
model = kplsr()
nlv = 0:30
gamma = (10).^(-5:1.:5)
pars = mpar(gamma = gamma)
rescv = gridcv(model, Xtrain, ytrain; segm,  score = rmsep, pars, nlv) ;
res = rescv.res 
lgamma = round.(log.(10, res.gamma), digits = 1)
plotgrid(res.nlv, res.y1, lgamma; step = 2, xlabel = "Nb. LVs",  ylabel = "RMSEP-CV", 
    leg_title = "Log(gamma)").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = kplsr(nlv = res.nlv[u], gamma = res.gamma[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f   

####---- Knnr 
nlvdis = [0; 15; 25] ; metric = [:sam]
h = [1, 2.5, 5]
k = [1; 5; 10; 20; 50 ; 100] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k)
length(pars[1]) 
model = knnr()
rescv = gridcv(model, Xtrain, ytrain; segm, score = rmsep, pars, verbose = true) ;
res = rescv.res 
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = knnr(nlvdis = res.nlvdis[u], metric = res.metric[u], h = res.h[u], k = res.k[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f   

####---- Lwplsr 
nlvdis = 15 ; metric = [:mah]
h = [1, 2, 5] ; k = [200, 350, 500] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k)
length(pars[1]) 
nlv = 0:20
model = lwplsr()
rescv = gridcv(model, Xtrain, ytrain; segm, score = rmsep, pars, nlv, verbose = true) ;
res = rescv.res 
group = string.("h=", res.h, " k=", res.k)
plotgrid(res.nlv, res.y1, group; xlabel = "Nb. LVs", ylabel = "RMSEP-CV").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = lwplsr(nlvdis = res.nlvdis[u], metric = res.metric[u], h = res.h[u], k = res.k[u], 
    nlv = res.nlv[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f   

####---- LwplsrAvg 
nlvdis = 15 ; metric = [:mah]
h = [1, 2, 5] ; k = [200, 350, 500] 
nlv = [0:20, 5:20] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k, nlv = nlv)
length(pars[1]) 
model = lwplsravg()
rescv = gridcv(model, Xtrain, ytrain; segm, score = rmsep, pars, verbose = true) ;
res = rescv.res 
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = lwplsravg(nlvdis = res.nlvdis[u], metric = res.metric[u], h = res.h[u], k = res.k[u], 
  nlv = res.nlv[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show rmsep(pred, ytest)
plotxy(pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f   
    
####------ Mbplsr
listbl = [1:525, 526:1050]
Xbltrain = mblock(Xtrain, listbl)
Xbltest = mblock(Xtest, listbl) 

model = mbplsr()
bscal = [:none, :frob]
pars = mpar(bscal = bscal) 
nlv = 0:30
rescv = gridcv(model, Xbltrain, ytrain; segm,  score = rmsep, pars, nlv) ;
res = rescv.res 
group = res.bscal 
plotgrid(res.nlv, res.y1, group; step = 2, xlabel = "Nb. LVs", ylabel = "RMSEP-CV").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = mbplsr(bscal = res.bscal[u], nlv = res.nlv[u])
fit!(model, Xbltrain, ytrain)
pred = predict(model, Xbltest).pred
@show rmsep(pred, ytest)
plotxy(pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f   
    
## Pipelines

####-- Pipeline Snv :> Savgol :> Plsr   (Only the last model is tuned)
## model1
model1 = snv(scal = false)
## model2 
npoint = 5 ; deriv = 0 ; degree = 2
model2 = savgol(; npoint, deriv, degree)
## model3
nlv = 0:30
model3 = plskern()
## Pipeline
model = pip(model1, model2, model3)
segm = segmkf(ntrain, 3; rep = 2)
res = gridcv(model, Xtrain, ytrain; segm, score = rmsep, nlv).res ;
@head res
plotgrid(res.nlv, res.y1; step = 2, xlabel = "Nb. LVs", ylabel = "RMSEP").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model3 = plskern(nlv = res.nlv[u])
model = pip(model1, model2, model3)
fit!(model, Xtrain, ytrain)
res = predict(model, Xtest) ; 
@head res.pred 
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction",
      ylabel = "Observed").f

####-- Pipeline Pca :> Svmr   (Only the last model is tuned)
## model1
nlv = 15 ; scal = true
model1 = pcasvd(; nlv, scal)
## model2
kern = [:krbf]
gamma = (10).^(-5:1.:5)
cost = (10).^(1:3)
epsilon = [.1, .2, .5]
pars = mpar(kern = kern, gamma = gamma, cost = cost, epsilon = epsilon)
model2 = svmr()
## Pipeline
model = pip(model1, model2)
res = gridcv(model, Xtrain, ytrain; segm, score = rmsep, pars, verbose = true).res ;
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model2 = svmr(kern = res.kern[u], gamma = res.gamma[u], cost = res.cost[u], epsilon = res.epsilon[u])
model = pip(model1, model2) 
fit!(model, Xtrain, ytrain)
res = predict(model, Xtest) ; 
@head res.pred 
rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", ylabel = "Observed").f

####### Discrimination

using Jchemo, JLD2, CairoMakie, JchemoData
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/forages2.jld2")
@load db dat
@names dat
X = dat.X
Y = dat.Y
tab(Y.typ)
s = Bool.(Y.test)
Xtrain = rmrow(X, s)
ytrain = rmrow(Y.typ, s)
Xtest = X[s, :]
ytest = Y.typ[s]
ntrain = nro(Xtrain)
ntest = nro(Xtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)

## a) Replicated K-fold CV 
K = 3 ; rep = 10
segm = segmkf(ntrain, K; rep)
## b) Replicated test-set validation
#m = round(Int, ntrain / 3) ; rep = 30
#segm = segmts(ntrain, m; rep)

####---- Plslda
model = plslda()
nlv = 1:30
pars = mpar(scal = [false; true])
rescv = gridcv(model, Xtrain, ytrain; segm, score = errp, pars, nlv)
res = rescv.res
typ = res.scal
plotgrid(res.nlv, res.y1, typ; step = 2, xlabel = "Nb. LVs", ylabel = "ERR").f
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
model = plslda(nlv = res.nlv[u], scal = res.scal[u])
fit!(model, Xtrain, ytrain)
pred = predict(model, Xtest).pred
@show errp(pred, ytest)
conf(pred, ytest).pct

## Computation of the confusion matrix within CV (average over the replications), for the best model 
matpred = Vector{Matrix{String}}(undef, rep * K)
k = 1
for i = 1:rep
    listsegm = segm[i]
    for j = 1:K
        s = listsegm[j]
        model = plslda(nlv = res.nlv[u], scal = res.scal[u])
        fit!(model, rmrow(Xtrain, s), rmrow(ytrain, s))
        pred = predict(model, Xtrain[s, :]).pred
        matpred[k] = hcat(pred, ytrain[s])
        k = k + 1
    end
end
respred = reduce(vcat, matpred)
conf(respred[:, 1], respred[:, 2]).pct

```
"""
function  gridcv(model, X, Y; segm::Vector{Vector{Vector{Int64}}}, 
        score::Function, pars::Union{Nothing, NamedTuple} = nothing, 
        nlv::Union{Nothing, Int, AbstractVector{Int}} = nothing, 
        lb::Union{Nothing, Float64, AbstractVector{Float64}} = nothing,  
        verbose::Bool = false) 
    q = nco(Y)
    nrep = length(segm)
    res_rep = list(nrep)
    @inbounds for i in eachindex(res_rep) 
        if verbose ; print("/ rep=", i, " ") ; end
        listsegm = segm[i]       # segments in the repetition
        nsegm = length(listsegm) # if segmts: = 1; if segmkf: = K
        zres = list(nsegm)       # results for the repetition
        @inbounds for j = 1:nsegm
            if verbose ; print("segm=", j, " ") ; end
            s = listsegm[j]
            ## Monoblock
            if isa(X[1, 1], Number)
                zres[j] = gridscore(model, rmrow(X, s), rmrow(Y, s), X[s, :], Y[s, :]; score, pars, nlv, lb)
            ## Multiblock
            else  
                Xcal = similar(X)
                Xval = similar(X)
                @inbounds for k in eachindex(X) 
                    Xcal[k] = rmrow(X[k], s)
                    Xval[k] = X[k][s, :]
                end
                zres[j] = gridscore(model, Xcal, rmrow(Y, s), Xval, Y[s, :]; score, pars, nlv, lb)
            end
        end
        ncomb = nro(zres[1])
        zres = reduce(vcat, zres)
        dat = DataFrame(rep = fill(i, nsegm * ncomb), segm = repeat(1:nsegm, inner = ncomb))
        zres = hcat(dat, zres)
        res_rep[i] = zres
    end
    if verbose ; println("/ End.") ; end
    res_rep = reduce(vcat, res_rep)
    ## Average scores over reps and segms
    if isnothing(nlv) && isnothing(lb)
        gdf = groupby(res_rep, collect(keys(pars))) 
    elseif !isnothing(nlv)
        namgroup = isnothing(pars) ? [:nlv] : [:nlv ; collect(keys(pars))]
        gdf = groupby(res_rep, namgroup) 
    elseif !isnothing(lb)
        namgroup = isnothing(pars) ? [:lb] : [:lb ; collect(keys(pars))]
        gdf = groupby(res_rep, namgroup) 
    end
    namy = map(string, fill("y", q), 1:q)
    res = combine(gdf, namy .=> meanv, renamecols = false)
    ## End
    (res = res, res_rep)
end

