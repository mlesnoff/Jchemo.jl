# To do: Add a structure a make a surchage of 'plot' of Makie 

"""
    iplsr(Xtrain, Ytrain, X, Y; nint = 5, score = rmsep, nlv,
        kwargs...)
Interval PLSR (iPLS) (Nørgaard et al. 2000)
* `Xtrain` : Training X-data (n, p).
* `Ytrain` : Training Y-data (n, q).
* `X` : Validation X-data (m, p).
* `Y` : Validation Y-data (m, q).
* `nint` : Nb. intervals. 
* `score` : Function computing the prediction score (= error rate; e.g. msep).
* `nlv` : Nb. latent variables (LVs) in the PLSR models.
- `kwarg` : Optional other arguments` to pass from funtion `plskern.

The range 1:p is segmented to `nint` column-intervals of equal (when possible) size. 
Then, the validation score is computed for each of the `nint` 
PLSR models and compared to the one fitted to the overal `Xtrain` (1:p).

## References
- Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J.P., Munck, L., 
Engelsen, S.B., 2000. Interval Partial Least-Squares Regression (iPLS): 
A Comparative Chemometric Study with an Example from Near-Infrared 
Spectroscopy. Appl Spectrosc 54, 413–419. https://doi.org/10.1366/0003702001949500

## Examples
```julia
using JchemoData, JLD2, CairoMakie, StatsBase
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "challenge2021_cal.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.tbc
n = nro(X)
wl = names(X)
wl_num = parse.(Float64, wl)

zX = -log.(10, Matrix(X)) 
f = 21 ; pol = 3 ; d = 2 ;
Xp = savgol(snv(X); f, pol, d) ;

nval = 200
s = sample(1:n, nval; replace = false)
Xcal = rmrow(Xp, s)
ycal = rmrow(y, s)
Xval = Xp[s, :]
yval = y[s]

res = iplsr(Xcal, ycal, Xval, yval; 
    nint = 10, nlv = 10) ;
zres = res.res
zres0 = res.res0
x = wl_num[zres.mid]
f, ax = scatter(x, zres.y1,
    axis = (xlabel = "Wawelength (nm)", ylabel = "RMSEP", xticks = x))
hlines!(ax, zres0.y1, linestyle = :dash)
f
```
"""
function iplsr(Xtrain, Ytrain, X, Y; 
        nint = 5, score = rmsep, nlv, kwargs...)
    Xtrain = ensure_mat(Xtrain)
    Ytrain = ensure_mat(Ytrain)
    X = ensure_mat(X)
    Y = ensure_mat(Y) 
    p = size(Xtrain, 2)
    q = size(Y, 2)
    z = collect(round.(range(1, p + 1, length = nint + 1)))
    int = [z[1:nint] z[2:(nint + 1)] .- 1]
    int = hcat(int, round.(rowmean(int)))
    int = Int64.(int)
    pred = similar(Y)
    res = list(nint, Matrix{Float64})
    @inbounds for i = 1:nint
        u = int[i, 1]:int[i, 2]
        fm = plskern(vcol(Xtrain, u), Ytrain; nlv = nlv, kwargs...)
        pred .= Jchemo.predict(fm, vcol(X, u)).pred
        res[i] = score(pred, Y)
    end
    res = reduce(vcat, res)
    fm = plskern(Xtrain, Ytrain; nlv = nlv)
    pred = Jchemo.predict(fm, X).pred
    res0 = score(pred, Y)
    dat = DataFrame(int, [:lo, :up, :mid])
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    res = hcat(dat, res)
    res0 = DataFrame(res0, Symbol.(namy))
    (res = res, res0 = res0, int = int)
end



