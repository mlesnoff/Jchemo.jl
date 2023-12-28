"""
    dmkern(X; h = nothing, a = 1)
Gaussian kernel density estimation (KDE).
* `X` : X-data (n, p).
* `h` : Define the bandwith, see examples
* `a` : Constant for the Scott's rule (default bandwith), 
    see thereafter.

Estimation of the probability density of `X` (column space) by non parametric
Gaussian kernels. 

Data `X` can be univariate (p = 1) or multivariate (p > 1). In the last 
case, function `dmkern` computes a multiplicative kernel such as in 
Scott & Sain 2005 Eq.19, and the internal bandwidth matrix `H` is diagonal
(see the code). **Note:  `H` in the code is often noted "H^(1/2)" in 
the litterature (e.g. Wikipedia).

The default bandwith is computed by:
* `h` = `a` * n^(-1 / (p + 4)) * colstd(`X`)
(`a` = 1 in Scott & Sain 2005).

## References 
Scott, D.W., Sain, S.R., 2005. 9 - Multidimensional Density Estimation, 
in: Rao, C.R., Wegman, E.J., Solka, J.L. (Eds.), Handbook of Statistics, 
Data Mining and Data Visualization. Elsevier, pp. 229–261. 
https://doi.org/10.1016/S0169-7161(04)24009-3

## Examples
```julia
using JLD2, CairoMakie

using JchemoData
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "iris.jld2") 
@load db dat
pnames(dat)
X = dat.X[:, 1:4] 
y = dat.X[:, 5]
n = nro(X)
tab(y) 

nlv = 2
fmda = fda(X, y; nlv = nlv) ;
pnames(fmda)
T = fmda.T
head(T)
n, p = size(T)

####  Probability density in the FDA score space (2D)

fm = dmkern(T) ;
pnames(fm)
fm.H
u = [1; 4; 150]
Jchemo.predict(fm, T[u, :]).pred

h = .3
fm = dmkern(T; h = h) ;
fm.H
u = [1; 4; 150]
Jchemo.predict(fm, T[u, :]).pred

h = [.3; .1]
fm = dmkern(T; h = h) ;
fm.H
u = [1; 4; 150]
Jchemo.predict(fm, T[u, :]).pred

## Bivariate distribution
npoints = 2^7
lims = [(minimum(T[:, j]), maximum(T[:, j])) for j = 1:nlv]
x1 = LinRange(lims[1][1], lims[1][2], npoints)
x2 = LinRange(lims[2][1], lims[2][2], npoints)
z = mpar(x1 = x1, x2 = x2)
grid = reduce(hcat, z)
m = nro(grid)
#plotxy(grid).f
fm = dmkern(T) ;
#fm = dmkern(T; a = .5) ;
#fm = dmkern(T; h = .3) ;
res = Jchemo.predict(fm, grid) ;
pred_grid = vec(res.pred)
f = Figure(size = (600, 400))
ax = Axis(f[1, 1]; 
    title = "Density for FDA scores (Iris)",
    xlabel = "Score 1", ylabel = "Score 2")
co = contour!(ax, grid[:, 1], grid[:, 2], pred_grid; 
    levels = 10, labels = true)
#Colorbar(f[1, 2], co; label = "Density")
scatter!(ax, T[:, 1], T[:, 2],
    color = :red, markersize = 5)
#xlims!(ax, -15, 15) ;ylims!(ax, -15, 15)
f

## Univariate distribution
x = T[:, 1]
fm = dmkern(x) ;
#fm = dmkern(x; a = .5) ;
#fm = dmkern(x; h = .3) ;
pred = Jchemo.predict(fm, x).pred 
f = Figure()
ax = Axis(f[1, 1])
hist!(ax, x; bins = 30, normalization = :pdf)  # area = 1
scatter!(ax, x, vec(pred);
    color = :red)
f

x = T[:, 1]
npoints = 2^8
lims = [minimum(x), maximum(x)]
#delta = 5 ; lims = [minimum(x) - delta, maximum(x) + delta]
grid = LinRange(lims[1], lims[2], npoints)
fm = dmkern(x) ;
#fm = dmkern(x; a = .5) ;
#fm = dmkern(x; h = .3) ;
pred_grid = Jchemo.predict(fm, grid).pred 
f = Figure()
ax = Axis(f[1, 1])
hist!(ax, x; bins = 30, normalization = :pdf)  # area = 1
lines!(ax, grid, vec(pred_grid); color = :red)
f
```
""" 
function dmkern(X; h = nothing, a = 1)
    X = ensure_mat(X)
    n, p = size(X)
    ## Particular case where n = 1
    ## (ad'hoc code for discrimination functions only)
    if n == 1
        H = diagm(repeat([a * n^(-1/(p + 4))], p))
    end
    ## End
    if isnothing(h)
        h = a * n^(-1 / (p + 4)) * colstd(X)      # a = .9, 1.06
        H = diagm(h)
    else 
        isa(h, Real) ? H = diagm(repeat([h], p)) : H = diagm(h)
    end
    Hinv = inv(H)
    detH = det(H)
    detH == 0 ? detH = 1e-20 : nothing
    Dmkern(X, H, Hinv, detH)
end

"""
    predict(object::Dmkern, x)
Compute predictions from a fitted model.
* `object` : The fitted model.
* `x` : Data (vector) for which predictions are computed.
""" 
function predict(object::Dmkern, X)
    X = ensure_mat(X)
    n, p = size(object.X)
    m = nro(X)
    pred = similar(X, m, 1)
    M = similar(object.X)
    @inbounds for i = 1:m
        M .= (vrow(X, i:i) .- object.X) * object.Hinv
    #Threads.@threads for i = 1:m
    #    M = (X[i:i, :] .- object.X) * object.Hinv
        sum2 = rowsum(M.^2)
        C = 1 / n * (2 * pi)^(-p / 2)
        pred[i, 1] =  C * (1 / object.detH) * sum(exp.(-.5 * sum2))
    end
    (pred = pred,)
end



