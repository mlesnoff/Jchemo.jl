"""
    isel!(model, X, Y, wl = 1:nco(X); score = rmsep, psamp = .3, nint = 5, rep = 1)
Interval variable selection.
* `model` : Model to evaluate.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `wl` : Optional numeric labels (p) of the X-columns.
Keyword arguments:  
* `score` : Function computing the prediction score (an error rate).
* `psamp` : Proportion of data used as validation set to compute the `score`.
* `nint` : Nb. intervals. 
* `rep` : Number of replications of the splitting calibration/validation (see below). 

The principle is as follows:
* Data (X, Y) are splitted randomly to a calibration set (Xcal, Ycal) and a validation set (Xval, Yval).
* The range 1:p in `X` is segmented to `nint` intervals of equal size (when possible). 
* The model is fitted on the calibration set and used to compute the predictions from Xval, firtsly accounting 
    for all the p variables (reference) and secondly for each (separately) of the `nint` intervals. The error 
    rates are computed by comparing the predictions to Yval. The interval-variable importance is the difference between 
    the reference error rate and the error rate computed for each intefval.

The overall process above is replicated `rep` times. The outputs provided by the function are the average 
results (i.e. over the `rep` replications;`imp`) and the results per replication (`res_rep`).

Note: the function is inplace (modifies object `model`).

## References
- Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J.V., Munck, L., Engelsen, S.B., 2000. Interval Partial 
Least-Squares Regression (iPLS): A Comparative Chemometric Study with an Example from Near-Infrared 
Spectroscopy. Appl Spectrosc 54, 413–419. https://doi.org/10.1366/0003702001949500

## Examples
```julia
using Jchemo, JchemoData, DataFrames, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "tecator.jld2") 
@load db dat
@names dat
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
res = isel!(model, Xtrain, ytrain, wl; nint, rep = 50) ;
dat = res.dat
res.imp 
res.res_rep

imp = res.imp[:, 1]
imp[imp .< 0] .= 0  # option: negative values are set to 0
lo = round.(dat.lo)
f = Figure(size = (650, 300))
ax = Axis(f[1, 1], xlabel = "Wawelength (nm)", ylabel = "Importance", xticks = lo)
scatter!(ax, dat.avg, imp; color = (:red, .5))
vlines!(ax, lo; color = :grey, linestyle = :dash, linewidth = 1)
hlines!(ax, [0]; color = :grey)
f
```
"""
function isel!(model, X, Y, wl = 1:nco(X); score = rmsep, psamp = .3, nint = 5, rep = 1)
    X = ensure_mat(X)
    Y = ensure_mat(Y) 
    n, p = size(X)
    q = nco(Y)
    nint = Int(nint)
    z = collect(round.(range(1, p + 1; length = nint + 1)))
    itv = [z[1:nint] z[2:(nint + 1)] .- 1]
    itv = hcat(itv, round.(rowmean(itv)))
    itv = Int.(itv)
    nval = round(Int, psamp * n)
    ncal = n - nval
    Xcal = similar(X, ncal, p)
    Ycal = similar(X, ncal, q)
    Xval = similar(X, nval, p)
    Yval = similar(X, nval, q)
    pred = similar(Yval)
    resref = zeros(1, q)   
    vres = list(Matrix{Float64}, nint)
    res_rep = zeros(nint, q, rep)
    @inbounds for i = 1:rep
        s = samprand(n, nval)
        Xcal .= vrow(X, s.train)
        Ycal .= vrow(Y, s.train)
        Xval .= vrow(X, s.test)
        Xval .= vrow(X, s.test)
        ## Reference
        fit!(model, Xcal, Ycal)
        pred .= predict(model, Xval).pred
        resref .= score(pred, Yval)
        ## Intervals
        @inbounds for j = 1:nint
            u = itv[j, 1]:itv[j, 2]
            fit!(model, vcol(Xcal, u), Ycal)
            pred .= predict(model, vcol(Xval, u)).pred
            vres[j] = resref - score(pred, Yval)
        end
        ## End
        res_rep[:, :, i] .= reduce(vcat, vres)
    end
    imp = mean(res_rep, dims = 3)[:, :, 1]
    dat = hcat(itv, )
    dat = DataFrame(itv, [:start, :end, :mid])
    dat.lo = wl[dat.start]
    dat.up = wl[dat.end]
    dat.avg = wl[dat.mid]
    (imp = imp, res_rep, dat)
end



