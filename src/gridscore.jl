"""
    gridscore(Xtrain, Ytrain, X, Y; score, fun, pars, verbose = FALSE) 
Model validation over a grid of parameters.
* `Xtrain` : Training X-data (n, p).
* `Ytrain` : Training Y-data (n, q).
* `X` : Validation X-data (m, p).
* `Y` : Validation Y-data (m, q).
* `score` : Function (e.g. `msep`) computing the prediction score.
* `fun` : Function computing the prediction model.
* `pars` : tuple of named vectors (= arguments of fun) of same length
    involved in the calculation of the score (e.g. output of function `mpar`).
* `verbose` : If true, fitting information are printed.

Compute a prediction score (= error rate) for a given model over a grid of parameters.

The score is computed over the validation sets `X` and `Y` for each combination 
of the grid defined in `pars`. 
    
The vectors in `pars` must have same length.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "cassav.jld2") 
@load db dat
pnames(dat)

# Building Train (years <= 2012) and Test  (year == 2012)

X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)
ntrain = nro(Xtrain)

# Building Cal and Val within Train

nval = 80
s = sample(1:ntrain, nval; replace = false)
Xcal = rmrow(Xtrain, s)
ycal = rmrow(ytrain, s)
Xval = Xtrain[s, :]
yval = ytrain[s]

# KNNR models

nlvdis = 15 ; metric = ["mahal" ]
h = [1 ; 2.5] ; k = [5 ; 10 ; 20 ; 50] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k) 
length(pars[1]) 
res = gridscore(Xcal, ycal, Xval, yval;
    score = rmsep, fun = knnr, pars = pars, verbose = true)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]

fm = knnr(Xtrain, ytrain;
    nlvdis = res.nlvdis[u], metric = res.metric[u],
    h = res.h[u], k = res.k[u]) ;
pred = Jchemo.predict(fm, Xtest).pred 
rmsep(pred, ytest)

################# PLSR models

nlv = 0:20
res = gridscorelv(Xcal, ycal, Xval, yval;
    score = rmsep, fun = plskern, nlv = nlv)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
plotgrid(res.nlv, res.y1;
    xlabel = "Nb. LVs", ylabel = "RMSEP").f

fm = plskern(Xtrain, ytrain; nlv = res.nlv[u]) ;
pred = Jchemo.predict(fm, Xtest).pred 
rmsep(pred, ytest)

# LWPLSR models

nlvdis = 15 ; metric = ["mahal" ]
h = [1 ; 2.5 ; 5] ; k = [50 ; 100] 
pars = mpar(nlvdis = nlvdis, metric = metric, h = h, k = k)
length(pars[1]) 
nlv = 0:20
res = gridscorelv(Xcal, ycal, Xval, yval;
    score = rmsep, fun = lwplsr, pars = pars, nlv = nlv, verbose = true)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
group = string.("h=", res.h, " k=", res.k)
plotgrid(res.nlv, res.y1, group;
    xlabel = "Nb. LVs", ylabel = "RMSECV").f

fm = lwplsr(Xtrain, ytrain;
    nlvdis = res.nlvdis[u], metric = res.metric[u],
    h = res.h[u], k = res.k[u], nlv = res.nlv[u]) ;
pred = Jchemo.predict(fm, Xtest).pred 
rmsep(pred, ytest)

################# RR models

lb = (10.).^collect(-5:1:-1)
res = gridscorelb(Xcal, ycal, Xval, yval;
    score = rmsep, fun = rr, lb = lb)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
plotgrid(log.(res.lb), res.y1;
    xlabel = "Lambda", ylabel = "RMSECV").f

fm = rr(Xtrain, ytrain; lb = res.lb[u]) ;
pred = Jchemo.predict(fm, Xtest).pred 
rmsep(pred, ytest)

################# KRR models

gamma = (10.).^collect(-4:1:4)
pars = mpar(gamma = gamma)
length(pars[1]) 
lb = (10.).^collect(-5:1:-1)
res = gridscorelb(Xcal, ycal, Xval, yval;
    score = rmsep, fun = krr, pars = pars, lb = lb)
u = findall(res.y1 .== minimum(res.y1))[1] 
res[u, :]
group = string.("gamma=", res.gamma)
plotgrid(log.(res.lb), res.y1, group;
    xlabel = "Lambda", ylabel = "RMSECV").f

fm = krr(Xtrain, ytrain; gamma = res.gamma[u], lb = res.lb[u]) ;
pred = Jchemo.predict(fm, Xtest).pred 
rmsep(pred, ytest)
```
"""
function gridscore(Xtrain, Ytrain, X, Y; score, fun, 
        pars, verbose = false)
    q = nco(Ytrain)
    ncomb = length(pars[1]) # nb. combinations in pars
    verbose ? println("-- Nb. combinations = ", ncomb) : nothing
    res = map(values(pars)...) do v...
        verbose ? println(Pair.(keys(pars), v)...) : nothing
        fm = fun(Xtrain, Ytrain ; Pair.(keys(pars), v)...)
        pred = Jchemo.predict(fm, X).pred
        score(pred, Y)
    end
    verbose ? println("-- End.") : nothing
    ncomb == 1 ? res = res[1] : res = reduce(vcat, res) 
    dat = DataFrame(pars)
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end

"""
    gridscorelv(Xtrain, Ytrain, X, Y; score, fun, nlv, pars, verbose = FALSE)
* See `gridscore`.
* `nlv` : Nb., or collection of nb., of latent variables (LVs).

Same as [`gridscore`](@ref) but specific to (and much faster for) models 
using latent variables (e.g. PLSR).

Argument `pars` must not contain `nlv`.

See `?gridscore` for examples.
"""
function gridscorelv(Xtrain, Ytrain, X, Y; score, fun, nlv, 
        pars = nothing, verbose = false)
    # If not multiblock
    if isa(Xtrain, Matrix)
        p = nco(Xtrain)
        nlv = max(0, minimum(nlv)):min(p, maximum(nlv))
    end
    # End
    q = nco(Ytrain)
    le_nlv = length(nlv)
    if isnothing(pars)
        verbose ? println("-- Nb. combinations = 0.") : nothing
        fm = fun(Xtrain, Ytrain, nlv = maximum(nlv))
        pred = Jchemo.predict(fm, X; nlv = nlv).pred
        le_nlv == 1 ? pred = [pred] : nothing
        res = zeros(le_nlv, q)
        @inbounds for i = 1:le_nlv
            res[i, :] = score(pred[i], Y)
        end
        dat = DataFrame(nlv = nlv)
    else       
        ncomb = length(pars[1])  # nb. combinations in pars
        verbose ? println("-- Nb. combinations = ", ncomb) : nothing
        res = map(values(pars)...) do v...    
            verbose ? println(Pair.(keys(pars), v)...) : nothing
            fm = fun(Xtrain, Ytrain ; nlv = maximum(nlv), Pair.(keys(pars), v)...)
            pred = Jchemo.predict(fm, X; nlv = nlv).pred
            le_nlv == 1 ? pred = [pred] : nothing
            zres = zeros(le_nlv, q)
            for i = 1:le_nlv
                zres[i, :] = score(pred[i], Y)
            end
            zres
        end 
        ncomb == 1 ? res = res[1] : res = reduce(vcat, res) 
        ## Make dat
        if le_nlv == 1
            dat = DataFrame(pars)
        else
            zdat = DataFrame(pars)
            dat = list(ncomb)
            @inbounds for i = 1:ncomb
                dat[i] = reduce(vcat, fill(zdat[i:i, :], le_nlv))
            end
            dat = reduce(vcat, dat)
        end
        znlv = repeat(nlv, ncomb)
        dat = hcat(dat, DataFrame(nlv = znlv))
        ## End
    end
    verbose ? println("-- End.") : nothing
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end

"""
    gridscorelb(Xtrain, Ytrain, X, Y; score, fun, lb, pars, verbose = FALSE)
* See `gridscore`.
* `lb` : Value, or collection of values, of the ridge regularization parameter "lambda".

Same as [`gridscore`](@ref) but specific to (and much faster for) models 
using ridge regularization (e.g. RR).

Argument `pars` must not contain `lb`.

See `?gridscore` for examples.
"""
function gridscorelb(Xtrain, Ytrain, X, Y; score, fun, lb, 
        pars = nothing, verbose = false)
    q = nco(Ytrain)
    lb = mlev(lb)
    le_lb = length(lb)
    if isnothing(pars)
        verbose ? println("-- Nb. combinations = 0.") : nothing
        fm = fun(Xtrain, Ytrain, lb = maximum(lb))
        pred = Jchemo.predict(fm, X; lb = lb).pred
        le_lb == 1 ? pred = [pred] : nothing
        res = zeros(le_lb, q)
        @inbounds for i = 1:le_lb
            res[i, :] = score(pred[i], Y)
        end
        dat = DataFrame(lb = lb)
    else
        ncomb = length(pars[1])  # nb. combinations in pars
        verbose ? println("-- Nb. combinations = ", ncomb) : nothing
        res = map(values(pars)...) do v...
            verbose ? println(Pair.(keys(pars), v)...) : nothing
            fm = fun(Xtrain, Ytrain ; lb = maximum(lb), Pair.(keys(pars), v)...)
            pred = Jchemo.predict(fm, X; lb = lb).pred
            le_lb == 1 ? pred = [pred] : nothing
            zres = zeros(le_lb, q)
            for i = 1:le_lb
                zres[i, :] = score(pred[i], Y)
            end
            zres
        end 
        ncomb == 1 ? res = res[1] : res = reduce(vcat, res) 
        ## Make dat
        if le_lb == 1
            dat = DataFrame(pars)
        else
            zdat = DataFrame(pars)
            dat = list(ncomb)
            @inbounds for i = 1:ncomb
                dat[i] = reduce(vcat, fill(zdat[i:i, :], le_lb))
            end
            dat = reduce(vcat, dat)
        end
        zlb = repeat(lb, ncomb)
        dat = hcat(dat, DataFrame(lb = zlb))
        ## End
    end
    verbose ? println("-- End.") : nothing
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    hcat(dat, res)
end

