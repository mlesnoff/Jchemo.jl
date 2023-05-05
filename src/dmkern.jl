struct Dmkern
    X::Array{Float64}
    H::Array{Float64}
    Hinv::Array{Float64}
    detH::Float64
end

"""
    dmkern(X; H = nothing, a = .5)
Gaussian kernel density estimation (KDE).
* `x` : Univariate data.
* `kwargs` : Optional arguments to pass in function `kde` of `KernelDensity.jl`.




## References 


## Examples
```julia
using CairoMakie

n = 10^4
x = randn(n)
lims = (minimum(x), maximum(x))
#lims = (-10, 10)
k = 2^3
bw = .1
fm = kde1(x; npoints = k, 
    bandwidth = bw,     # default: Silverman's rule
    #boundary = lims    # default: See KernelDensity
    ) ;
fm.x
diff(fm.x)
ds = fm.density    # densities with same scale as ':pdf' in Makie.hist
sum(ds * diff(fm.x)[1])  # = 1
## Normalization to sum to 1
dstot = sum(ds)
dsn = ds / dstot 
sum(dsn)
## Standardization to a uniform distribution
## (if > 1 ==> "dense" areas) 
mu_unif = mean(ds)
ds / mu_unif 

## Prediction
xnew = [-200; -100; -1; 0; 1; 200]
pred = Jchemo.predict(fm, xnew).pred    # densities
pred / dstot 
pred / mu_unif 

n = 10^3 
x = randn(n)
lims = (minimum(x), maximum(x))
#lims = (-6, 6)
k = 2^8
bw = .1
fm = kde1(x; npoints = k, 
    bandwidth = bw,         # Default = Silverman's rule
    boundary = lims
    ) ;
f = Figure(resolution = (500, 350))
ax = Axis(f[1, 1];
    xlabel = "x", ylabel = "Density")
hist!(ax, x; bins = 30, normalization = :pdf)  # area = 1
lines!(ax, fm.x, fm.density;
    color = :red)
f
```
""" 
function dmkern(X; H = nothing, a = .5)
    X = ensure_mat(X)
    n, p = size(X)
    ## Case where n = 1
    ## (ad'hoc for discrimination functions only)
    if n == 1
        H = diagm(repeat([a * n^(-1/(p + 4))], p))
    end
    ## End
    if isnothing(H)
        h = a * n^(-1 / (p + 4)) * colstd(X)      # a = .9, 1.06
        H = diagm(h)
    else 
        isa(H, Real) ? H = diagm(repeat([H], p)) : nothing
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
        sum2 = rowsum(M.^2)
        pred[i, 1] = 1 / n * (2 * pi)^(-p / 2) * (1 / object.detH) * sum(exp.(-.5 * sum2))
    end
    (pred = pred,)
end



