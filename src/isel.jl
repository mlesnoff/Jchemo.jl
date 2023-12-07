"""
    isel(X, Y, wl = 1:nco(X); rep = 1, 
        nint = 5, psamp = 1/3, score = rmsep, 
        fun, kwargs...)
Interval variable selection.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `wl` : Optional numeric labels (p, 1) of the X-columns.  
* `rep` : Number of replications. 
* `nint` : Nb. intervals. 
* `psamp` : Proportion of data used as test set to compute the `score`
    (default: n/3 of the data).
* `score` : Function computing the prediction score (= error rate; e.g. msep).
* `fun` : Function defining the prediction model.
* `kwarg` : Optional other arguments to pass to funtion defined in `fun`.

The principle is as follows:
* Data (X, Y) are splitted randomly to a training and a test set.
* Range 1:p in `X` is segmented to `nint` intervals of equal (when possible) size. 
* The model is fitted on the training set and the score (error rate) on the test set, 
    firtsly accounting for all the p variables (reference) and secondly 
    for each of the `nint` intervals. 
* This process is replicated `rep` times. Average results are provided in the outputs,
    as well the results per replication. 

## References
- Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J.P., Munck, L., 
Engelsen, S.B., 2000. Interval Partial Least-Squares Regression (iPLS): 
A Comparative Chemometric Study with an Example from Near-Infrared 
Spectroscopy. Appl Spectrosc 54, 413–419. https://doi.org/10.1366/0003702001949500

## Examples
```julia
using JchemoData, DataFrames, JLD2
using CairoMakie

path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/tecator.jld2") 
@load db dat
pnames(dat)

X = dat.X
Y = dat.Y 
wlstr = names(X)
wl = parse.(Float64, wlstr) 
typ = Y.typ
y = Y.fat

f = 15 ; pol = 3 ; d = 2 
Xp = savgol(snv(X); f = f, pol = pol, d = d) 

s = typ .== "train"
Xtrain = Xp[s, :]
ytrain = y[s]

nint = 10
nlv = 5
res = isel(Xtrain, ytrain, wl; rep = 20, 
    nint = nint, fun = plskern, nlv = nlv) ;
res.res_rep
res.res0_rep
zres = res.res
zres0 = res.res0

f = Figure(size = (900, 400))
ax = Axis(f[1, 1],
    xlabel = "Wawelength (nm)", ylabel = "RMSEP",
    xticks = zres.lo)
scatter!(ax, zres.mid, zres.y1; color = (:red, .5))
vlines!(ax, zres.lo; color = :grey,
    linestyle = :dash, linewidth = 1)
hlines!(ax, zres0.y1, linestyle = :dash)
f
```
"""
function isel(X, Y, wl = 1:nco(X); rep = 1, 
        nint = 5, psamp = 1/3, score = rmsep, 
        fun, kwargs...)
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
    s = list(nval, Int)
    res0_rep = zeros(1, q, rep)   
    zres = list(nint, Matrix{Float64})
    res_rep = zeros(nint, q, rep)
    @inbounds for i = 1:rep
        s .= sample(1:n, nval; replace = false)
        Xcal .= rmrow(X, s)
        Ycal .= rmrow(Y, s)
        Xval .= X[s, :]
        Yval .= Y[s, :]
        ## All variables (Ref0)
        fm = fun(Xcal, Ycal; kwargs...)
        pred = Jchemo.predict(fm, Xval).pred
        res0_rep[:, :, i] = score(pred, Yval)
        ## Intervals
        @inbounds for j = 1:nint
            u = int[j, 1]:int[j, 2]
            fm = fun(vcol(Xcal, u), Ycal; kwargs...)
            pred .= Jchemo.predict(fm, vcol(Xval, u)).pred
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



