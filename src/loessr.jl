"""
    loessr(X, y; kwargs...)
Compute a locally weighted regression model (LOESS).
* `X` : X-data (n, p).
* `y` : Univariate y-data (n).
Keyword arguments:
* `span` : Window for neighborhood selection (level of smoothing)
    for the local fitting, typically in [0, 1] (proportion).
* `degree` : Polynomial degree for the local fitting.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.
    
The function fits a LOESS model using package `Loess.jl'. 

Smaller values of `span` result in smaller local context in fitting (less smoothing).

## References

https://github.com/JuliaStats/Loess.jl

Cleveland, W. S. (1979). Robust locally weighted regression and smoothing 
scatterplots. Journal of the American statistical association, 74(368), 829-836. 
DOI: 10.1080/01621459.1979.10481038

Cleveland, W. S., & Devlin, S. J. (1988). Locally weighted regression: an approach 
to regression analysis by local fitting. Journal of the American statistical association, 
83(403), 596-610. DOI: 10.1080/01621459.1988.10478639

Cleveland, W. S., & Grosse, E. (1991). Computational methods for local regression. 
Statistics and computing, 1(1), 47-62. DOI: 10.1007/BF01890836

## Examples
```julia
using Jchemo, CairoMakie

####### Example of fitting the function sinc(x)
####### described in Rosipal & Trejo 2001 p. 105-106 
x = collect(-10:.2:10) 
x[x .== 0] .= 1e-5
n = length(x)
zy = sin.(abs.(x)) ./ abs.(x) 
y = zy + .2 * randn(n) 
model = mod_(loessr; span = 1 / 3) 
fit!(model, x, y)
pred = predict(model, x).pred 
f = Figure(size = (700, 300))
ax = Axis(f[1, 1], xlabel = "x", ylabel = "y")
scatter!(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred); label = "Loess")
f[1, 2] = Legend(f, ax, framevisible = false)
f
```
""" 
function loessr(X, y; kwargs...)
    par = recovkw(ParLoessr, kwargs).par
    X = ensure_mat(X)
    Q = eltype(X)
    y = vec(y)
    p = nco(X)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X)
        X = fscale(X, xscales)
    end
    fm = Loess.loess(X, y; span = par.span, degree = par.degree) 
    Loessr(fm, xscales, par) 
end

"""
    predict(object::Loessr, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Loessr, X)
    X = ensure_mat(X)
    m = nro(X)
    Q = eltype(X)
    pred = Loess.predict(object.fm, fscale(X, object.xscales))
    pred = reshape(convert.(Q, pred), m, 1)
    (pred = pred,)
end

