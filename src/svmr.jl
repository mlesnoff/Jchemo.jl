"""
    svmr(X, y; kern = :krbf, 
        gamma = 1. / size(X, 2), degree = 3, coef0 = 0., cost = 1., 
        epsilon = .1, scal = false)
Support vector machine for regression (Epsilon-SVR).
* `X` : X-data (n, p).
* `y` : Univariate y-data (n).
Keyword arguments:
* `kern` : Type of kernel used to compute the Gram matrices.
    Possible values are: `:krbf`, `:kpol`. See respective 
    functions `krbf` and `kpol` for their keyword arguments.

* `kern` : Type of kernel used to compute the Gram matrices.
    Possible values are :krbf, :kpol, :klin, :ktanh". 
* `gamma` : `kern` parameter, see below.
* `degree` : `kern` parameter, see below.
* `coef0` : `kern` parameter, see below.
* `cost` : Cost of constraints violation C parameter.
* `epsilon` : Epsilon parameter in the loss function.
* `scal` : Boolean. If `true`, each column of `X` 
    is scaled by its uncorrected standard deviation.

Kernel types: 
* :krbf -- radial basis function: exp(-gamma * ||x - y||^2)
* :kpol -- polynomial: (gamma * x' * y + coef0)^degree
* "klin -- linear: x' * y
* :ktan -- sigmoid: tanh(gamma * x' * y + coef0)

The function uses LIBSVM.jl (https://github.com/JuliaML/LIBSVM.jl) 
that is an interface to library LIBSVM (Chang & Li 2001).

## References 

Julia package LIBSVM.jl: https://github.com/JuliaML/LIBSVM.jl

Chang, C.-C. & Lin, C.-J. (2001). LIBSVM: a library for support 
vector machines. Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm. 
Detailed documentation (algorithms, formulae, ...) can be found in
http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.ps.gz

Chih-Chung Chang and Chih-Jen Lin, LIBSVM: a library for support 
vector machines. ACM Transactions on Intelligent Systems and 
Technology, 2:27:1--27:27, 2011. Software available at 
http://www.csie.ntu.edu.tw/~cjlin/libsvm

Schölkopf, B., Smola, A.J., 2002. Learning with kernels: 
support vector machines, regularization, optimization, and 
beyond. Adaptive computation and machine learning. MIT Press, 
Cambridge, Mass.

## Examples
```julia
using JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
pnames(dat)
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)
p = nco(X)

kern = :krbf ; gamma = .1
cost = 1000 ; epsilon = 1
mod = svmr(; kern, gamma,
    cost, epsilon) ;
fit!(mod, Xtrain, ytrain)
pnames(mod)
pnames(mod.fm)

res = predict(mod, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5),
    bisect = true, xlabel = "Prediction", 
    ylabel = "Observed").f    

####### Example of fitting the function sinc(x)
####### described in Rosipal & Trejo 2001 p. 105-106 
x = collect(-10:.2:10) 
x[x .== 0] .= 1e-5
n = length(x)
zy = sin.(abs.(x)) ./ abs.(x) 
y = zy + .2 * randn(n) 
kern = :krbf
gamma = .1
mod = svmr(; kern, gamma) ;
fit!(mod, x, y)
pred = predict(mod, x).pred 
f, ax = scatter(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "Fitted model")
axislegend("Method")
f
```
""" 
function svmr(X, y; kwargs...)
    par = recovkwargs(Par, kwargs)
    kern = par.kern 
    @assert in([:krbf, :kpol, :klin, :ktanh])(kern) "Wrong value for argument 'kern'." 
    X = ensure_mat(X)
    Q = eltype(X)
    y = vec(y)
    p = nco(X)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X)
        X = fscale(X, xscales)
    end
    if kern == :krbf
        fkern = LIBSVM.Kernel.RadialBasis
    elseif kern == :kpol
        fkern = LIBSVM.Kernel.Polynomial
    elseif kern == :klin
        fkern = LIBSVM.Kernel.Linear
    elseif kern == :ktanh
        fkern = LIBSVM.Kernel.Sigmoid
    end
    fm = svmtrain(X', y;
        svmtype = EpsilonSVR, 
        kernel = fkern,
        gamma =  par.gamma,
        coef0 = par.coef0,
        degree = par.degree,
        cost = par.cost, 
        epsilon = par.epsilon,
        tolerance = 0.001,
        nt = 0,
        verbose = false) 
    Svmr(fm, xscales)
end

"""
    predict(object::Svmr, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Svmr, X)
    X = ensure_mat(X)
    Q = eltype(X)
    pred = svmpredict(object.fm, 
        fscale(X, object.xscales)')[1]
    m = length(pred)
    pred = reshape(convert.(Q, pred), m, 1)
    (pred = pred,)
end

