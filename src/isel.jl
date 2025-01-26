"""
    isel!(model, X, Y, wl = 1:nco(X); rep = 1, nint = 5, psamp = .3, score = rmsep)
Interval variable selection.
* `model` : Model to evaluate.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `wl` : Optional numeric labels (p, 1) of the X-columns.
Keyword arguments:  
* `rep` : Number of replications of the splitting
    training/test. 
* `nint` : Nb. intervals. 
* `psamp` : Proportion of data used as test set 
    to compute the `score`.
* `score` : Function computing the prediction score.

The principle is as follows:
* Data (X, Y) are splitted randomly to a training and a test set.
* Range 1:p in `X` is segmented to `nint` intervals, when possible 
    of equal size. 
* The model is fitted on the training set and the score (error rate) 
    on the test set, firtsly accounting for all the p variables 
    (reference) and secondly for each of the `nint` intervals. 
* This process is replicated `rep` times. Average results are provided 
    in the outputs, as well the results per replication. 

## References
- Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J.V., Munck, L., 
Engelsen, S.B., 2000. Interval Partial Least-Squares Regression (iPLS): 
A Comparative Chemometric Study with an Example from Near-Infrared 
Spectroscopy. Appl Spectrosc 54, 413–419. 
https://doi.org/10.1366/0003702001949500

## Examples
```julia
using Jchemo, JchemoData, DataFrames, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "tecator.jld2") 
@load db dat
pnames(dat)
X = dat.X
Y = dat.Y 
wl_str = names(X)
wl = parse.(Float64, wl_str) 
ntot, p = size(X)
typ = Y.typ
namy = names(Y)[1:3]
plotsp(X, wl; xlabel = "Wavelength (nm)", ylabel = "Absorbance").f

s = typ .== "train"
Xtrain = X[s, :]
Ytrain = Y[s, namy]
Xtest = rmrow(X, s)
Ytest = rmrow(Y[:, namy], s)
ntrain = nro(Xtrain)
ntest = nro(Xtest)
ntot = ntrain + ntest
(ntot = ntot, ntrain, ntest)

## Work on the j-th y-variable 
j = 2
nam = namy[j]
ytrain = Ytrain[:, nam]
ytest = Ytest[:, nam]

model = plskern(nlv = 5)
nint = 10
res = isel!(model, Xtrain, ytrain, wl; rep = 30, nint) ;
res.res_rep
res.res0_rep
zres = res.res
zres0 = res.res0
f = Figure(size = (650, 300))
ax = Axis(f[1, 1], xlabel = "Wawelength (nm)", ylabel = "RMSEP_Val",
    xticks = zres.lo)
scatter!(ax, zres.mid, zres.y1; color = (:red, .5))
vlines!(ax, zres.lo; color = :grey, linestyle = :dash, linewidth = 1)
hlines!(ax, zres0.y1, linestyle = :dash)
f
```
"""
function isel!(model, X, Y, wl = 1:nco(X); rep = 1, nint = 5, psamp = .3, score = rmsep)
    X = ensure_mat(X)
    Y = ensure_mat(Y) 
    n, p = size(X)
    q = nco(Y)
    nint = Int(nint)
    z = collect(round.(range(1, p + 1; length = nint + 1)))
    int = [z[1:nint] z[2:(nint + 1)] .- 1]
    int = hcat(int, round.(rowmean(int)))
    int = Int.(int)
    nval = Int(round(psamp * n))
    ncal = n - nval
    Xcal = similar(X, ncal, p)
    Ycal = similar(X, ncal, q)
    Xval = similar(X, nval, p)
    Yval = similar(X, nval, q)
    s = list(Int, nval)
    res0_rep = zeros(1, q, rep)   
    zres = list(Matrix{Float64}, nint)
    res_rep = zeros(nint, q, rep)
    @inbounds for i = 1:rep
        s = samprand(n, nval)
        Xcal .= vrow(X, s.train)
        Ycal .= vrow(Y, s.train)
        Xval .= vrow(X, s.test)
        Xval .= vrow(X, s.test)
        ## All variables ('res0')
        fit!(model, Xcal, Ycal)
        pred = predict(model, Xval).pred
        res0_rep[:, :, i] = score(pred, Yval)
        ## Intervals
        @inbounds for j = 1:nint
            u = int[j, 1]:int[j, 2]
            fit!(model, vcol(Xcal, u), Ycal)
            pred = predict(model, vcol(Xval, u)).pred
            zres[j] = score(pred, Yval)
        end
        ## End
        res_rep[:, :, i] .= reduce(vcat, zres)
    end
    dat = DataFrame(int, [:lo, :up, :mid])
    dat.lo = wl[dat.lo]
    dat.up = wl[dat.up]
    dat.mid = wl[dat.mid]
    namy = map(string, repeat(["y"], q), 1:q)
    res = mean(res_rep, dims = 3)[:, :, 1]
    res = DataFrame(res, Symbol.(namy))
    res = hcat(dat, res)
    res0 = mean(res0_rep, dims = 3)[:, :, 1]
    res0 = DataFrame(res0, Symbol.(namy))
    (res = res, res0, res_rep, res0_rep, int)
end



