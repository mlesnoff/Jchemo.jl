"""
    dmkern(X; kwargs...)
Gaussian kernel density estimation (KDE).
* `X` : X-data (n, p).
Keyword arguments:
* `h_kde` : Define the bandwith, see examples.
* `a_kde` : Constant for the Scott's rule 
    (default bandwith), see thereafter.

Estimation of the probability density of `X` (column space) by 
non parametric Gaussian kernels. 

Data `X` can be univariate (p = 1) or multivariate (p > 1). 
In the last case, function `dmkern` computes a multiplicative 
kernel such as in Scott & Sain 2005 Eq.19, and the internal bandwidth 
matrix `H` is diagonal (see the code). 

**Note:**  `H` in the `dmkern` code is often noted "H^(1/2)" in the 
litterature (e.g. Wikipedia).

The default bandwith is computed by:
* `h_kde` = `a_kde` * n^(-1 / (p + 4)) * colstd(`X`)
(`a_kde` = 1 in Scott & Sain 2005).

## References 
Scott, D.W., Sain, S.R., 2005. 9 - Multidimensional Density 
Estimation, in: Rao, C.R., Wegman, E.J., Solka, J.L. (Eds.), 
Handbook of Statistics, Data Mining and Data Visualization. 
Elsevier, pp. 229–261. 
https://doi.org/10.1016/S0169-7161(04)24009-3

## Examples
```julia
using JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "iris.jld2") 
@load db dat
pnames(dat)
X = dat.X[:, 1:4] 
y = dat.X[:, 5]
n = nro(X)
tab(y) 

mod = model(fda; nlv = 2)
fit!(mod, X, y)
@head T = mod.fm.T
p = nco(T)

#### Probability density in the FDA 
#### score space (2D)

mod = model(dmkern)
fit!(mod, T) 
pnames(mod.fm)
mod.fm.H
u = [1; 4; 150]
predict(mod, T[u, :]).pred

h_kde = .3
mod = model(dmkern; h_kde)
fit!(mod, T) 
mod.fm.H
u = [1; 4; 150]
predict(mod, T[u, :]).pred

h_kde = [.3; .1]
mod = model(dmkern; h_kde)
fit!(mod, T) 
mod.fm.H
u = [1; 4; 150]
predict(mod, T[u, :]).pred

## Bivariate distribution
npoints = 2^7
nlv = 2
lims = [(minimum(T[:, j]), maximum(T[:, j])) for j = 1:nlv]
x1 = LinRange(lims[1][1], lims[1][2], npoints)
x2 = LinRange(lims[2][1], lims[2][2], npoints)
z = mpar(x1 = x1, x2 = x2)
grid = reduce(hcat, z)
m = nro(grid)
mod = model(dmkern) 
#mod = model(dmkern; a_kde = .5) 
#mod = model(dmkern; h_kde = .3) 
fit!(mod, T) 

res = predict(mod, grid) ;
pred_grid = vec(res.pred)
f = Figure(size = (600, 400))
ax = Axis(f[1, 1];  title = "Density for FDA scores (Iris)", xlabel = "Score 1", 
    ylabel = "Score 2")
co = contour!(ax, grid[:, 1], grid[:, 2], pred_grid; levels = 10, labels = true)
scatter!(ax, T[:, 1], T[:, 2], color = :red, markersize = 5)
#xlims!(ax, -15, 15) ;ylims!(ax, -15, 15)
f

## Univariate distribution
x = T[:, 1]
mod = model(dmkern) 
#mod = model(dmkern; a_kde = .5) 
#mod = model(dmkern; h_kde = .3) 
fit!(mod, x) 
pred = predict(mod, x).pred 
f = Figure()
ax = Axis(f[1, 1])
hist!(ax, x; bins = 30, normalization = :pdf)  # area = 1
scatter!(ax, x, vec(pred); color = :red)
f

x = T[:, 1]
npoints = 2^8
lims = [minimum(x), maximum(x)]
#delta = 5 ; lims = [minimum(x) - delta, maximum(x) + delta]
grid = LinRange(lims[1], lims[2], npoints)
mod = model(dmkern) 
#mod = model(dmkern; a_kde = .5) 
#mod = model(dmkern; h_kde = .3) 
fit!(mod, x) 
pred_grid = predict(mod, grid).pred 
f = Figure()
ax = Axis(f[1, 1])
hist!(ax, x; bins = 30, normalization = :pdf)  # area = 1
lines!(ax, grid, vec(pred_grid); color = :red)
f
```
""" 
function dmkern(X; kwargs...)
    par = recovkwargs(Par, kwargs)
    X = ensure_mat(X)
    n, p = size(X)
    h_kde = par.h_kde
    a_kde = par.a_kde
    ## Particular case where n = 1
    ## (ad'hoc code for discrimination functions only)
    if n == 1
        H = diagm(repeat([a_kde * n^(-1/(p + 4))], p))
    end
    ## End
    if isnothing(h_kde)
        h_kde = a_kde * n^(-1 / (p + 4)) * colstd(X)      # a_kde = .9, 1.06
        H = diagm(h_kde)
    else 
        isa(h_kde, Real) ? H = diagm(repeat([h_kde], p)) : H = diagm(h_kde)
    end
    Hinv = inv(H)
    detH = det(H)
    detH == 0 ? detH = 1e-20 : nothing
    Dmkern(X, H, Hinv, detH)
end

"""
    predict(object::Dmkern, x)
Compute predictions from a_kde fitted model.
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



