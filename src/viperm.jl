"""
    viperm(X, Y; perm = 50,
        psamp = .3, score = rmsep, fun, kwargs...)
Variable importance by direct permutations.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).  
* `perm` : Number of replications. 
* `nint` : Nb. intervals. 
* `psamp` : Proportion of data used as test set to compute the `score`
    (default: 30% of the data).
* `score` : Function computing the prediction score (= error rate; e.g. msep).
* `fun` : Function defining the prediction model.
* `kwarg` : Optional other arguments to pass to funtion defined in `fun`.

The principle is as follows:
* Data (X, Y) are splitted randomly to a training and a test set.
* The model is fitted on Xtrain, and the score (error rate) on Xtest.
    This gives the reference error rate.
* Rows of a given variable (feature) j in Xtest are randomly permutated
    (the rest of Xtest is unchanged). The score is computed on 
    the permuted Xtest and the new score is computed. The importance
    is computed by the difference between this score and the reference score.
* This process is run for each variable separately and replicated `perm` times.
    Average results are provided in the outputs, as well the results per 
    replication. 

In general, this method returns similar results as the out-of-bag permutation method
used in random forests (Breiman, 2001).

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

res = viperm(Xtrain, ytrain; perm = 50, 
    score = rmsep, fun = plskern, nlv = 9)
f = Figure(resolution = (500, 400))
ax = Axis(f[1, 1];
    xlabel = "Wavelength (nm)", 
    ylabel = "Importance")
scatter!(ax, wl, vec(res.imp); color = (:red, .5))
u = [910; 950]
vlines!(ax, u; color = :grey, linewidth = 1)
f
```
"""
function viperm(X, Y; perm = 50,
        psamp = .3, score = rmsep, fun, kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y) 
    n, p = size(X)
    q = nco(Y)
    nval = Int(round(psamp * n))
    ncal = n - nval
    Xcal = similar(X, ncal, p)
    Ycal = similar(X, ncal, q)
    Xval = similar(X, nval, p)
    Yval = similar(X, nval, q)
    s = list(nval, Int)
    res = similar(X, p, q, perm)
    @inbounds for i = 1:perm
        s .= sample(1:n, nval; replace = false)  
        Xcal .= rmrow(X, s)
        Ycal .= rmrow(Y, s)
        Xval .= X[s, :]
        Yval .= Y[s, :]
        fm = fun(Xcal, Ycal; kwargs...)
        pred = predict(fm, Xval).pred
        score0 = score(pred, Yval)
        zXval = similar(Xval)
        @inbounds for j = 1:p
            zXval .= copy(Xval)
            zs = sample(1:nval, nval, replace = false)
            zXval[:, j] .= zXval[zs, j]
            pred .= predict(fm, zXval).pred
            zscore = score(pred, Yval)
            res[j, :, i] = zscore .- score0
        end
    end
    imp = reshape(mean(res, dims = 3), p, q)
    (imp = imp, res)
end 
