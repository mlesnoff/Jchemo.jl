"""
    viperm(model, X, Y; rep = 50, psamp = .3, score = rmsep)
Variable importance by direct permutations.
* `model` : Model to evaluate.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).  
Keyword arguments:
* `rep` : Number of replications of the splitting training/test. 
* `psamp` : Proportion of data used as test set to compute the `score`.
* `score` : Function computing the prediction score (an error rate).

The principle is as follows:
* Data (X, Y) are splitted randomly to a calibration set (Xcal, Ycal) and a validation set (Xval, Yval).
* The model is fitted on (Xcal, Ycal) and used to compute the predictions from Xval. The error rate (the `score`) 
    is computed by comparing these predictions to Yval, giving the reference error rate.
* Consider Xval and a given variable (feature = column) j.
    * a) The rows of variable j are permutated randomly (the other columns of Xval are unchanged), generating a new
        matrix "Xval.perm.j".
    * b) The model is used to compute the predictions from Xval.perm.j, and the error rate is computed by comparing 
        these predictions to Yval. The variable importance (for variable j) is the difference between this error rate and 
        the reference error rate.
* This process is run for each variable j, separately.

The overall process above is replicated `rep` times. The output provided by the function contains the average 
results (i.e. over the `rep` replications) and the results per replication.

In general, this method returns similar results as the out-of-bag permutation method (such as the one used in random 
forests; Breiman, 2001).

## References
Breiman, L., 2001. Random Forests. Machine Learning 45, 5–32. https://doi.org/10.1023/A:1010933404324

Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J.V., Munck, L., Engelsen, S.B., 2000. Interval Partial 
Least-Squares Regression (iPLS): A Comparative Chemometric Study with an Example from Near-Infrared 
Spectroscopy. Appl Spectrosc 54, 413–419. https://doi.org/10.1366/0003702001949500

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
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

model = plskern(nlv = 9)
res = viperm(model, Xtrain, ytrain; rep = 50, score = rmsep) ;
z = vec(res.imp)
f = Figure(size = (500, 400))
ax = Axis(f[1, 1]; xlabel = "Wavelength (nm)", ylabel = "Importance")
scatter!(ax, wl, vec(z); color = (:red, .5))
u = [910; 950]
vlines!(ax, u; color = :grey, linewidth = 1)
f

model = rfr(n_trees = 10, max_depth = 2000, min_samples_leaf = 5)
res = viperm(model, Xtrain, ytrain; rep = 50)
z = vec(res.imp)
f = Figure(size = (500, 400))
ax = Axis(f[1, 1]; xlabel = "Wavelength (nm)", ylabel = "Importance")
scatter!(ax, wl, vec(z); color = (:red, .5))
u = [910; 950]
vlines!(ax, u; color = :grey, linewidth = 1)
f
```
"""
function viperm(model, X, Y; rep = 50, psamp = .3, score = rmsep)
    X = ensure_mat(X)
    Y = ensure_mat(Y) 
    n, p = size(X)
    q = nco(Y)
    nval = round(Int, psamp * n)
    ncal = n - nval
    Xcal = similar(X, ncal, p)
    Ycal = similar(X, ncal, q)
    Xval = similar(X, nval, p)
    Yval = similar(X, nval, q)
    s = list(Int, nval)
    res = similar(X, p, q, rep)
    @inbounds for i = 1:rep
        s = samprand(n, nval)
        Xcal .= X[s.train, :]
        Ycal .= Y[s.train, :]
        Xval .= X[s.test, :]
        Yval .= Y[s.test, :]
        fit!(model, Xcal, Ycal)
        pred = predict(model, Xval).pred
        score0 = score(pred, Yval)
        zXval = similar(Xval)
        @inbounds for j = 1:p
            zXval .= copy(Xval)
            ## Permutation variable j
            zs = StatsBase.sample(1:nval, nval, replace = false)
            ## End  
            zXval[:, j] .= zXval[zs, j]
            pred .= predict(model, zXval).pred
            zscore = score(pred, Yval)
            res[j, :, i] = zscore .- score0
        end
    end
    imp = reshape(mean(res, dims = 3), p, q)
    (imp = imp, res)
end 
