struct Svmr
    fm
end

"""
    svmr(X, y; kern = "rbf", 
        gamma = 1. / size(X, 2), degree = 3, coef0 = 0., cost = 1., 
        epsilon = .1)
Support vector machine for regression (Epsilon-SVR).
* `X` : X-data.
* `y : y-data (univariate).
* 'kern' : Type of kernel used to compute the Gram matrices.
    Possible values are "krbf", "kpol", "klin" or "ktanh". 
* 'gamma' : See below.
* 'degree' : See below.
* 'coef0' : See below.
* 'cost' : Cost of constraints violation C parameter.
* 'epsilon' : Epsilon parameter in the loss function .

Kernel types : 
* "krbf" -- radial basis function: exp(-gamma * |x - y|^2)
* "kpol" -- polynomial: (gamma * x' * y + coef0)^degree
* "klin* -- linear: x' * y
* "ktan" -- sigmoid: tanh(gamma * x' * y + coef0)

The function uses LIBSVM.jl (https://github.com/JuliaML/LIBSVM.jl) 
that is an interface to library LIBSVM (Chang & Li 2001).

## References 

Julia package LIBSVM.jl: https://github.com/JuliaML/LIBSVM.jl

Chang, C.-C. & Lin, C.-J. (2001). LIBSVM: a library for support vector machines. 
Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm. 
Detailed documentation (algorithms, formulae, ...) can be found in
http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.ps.gz

Chih-Chung Chang and Chih-Jen Lin, LIBSVM: a library for support vector machines. 
ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011. 
Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm

Schölkopf, B., Smola, A.J., 2002. Learning with kernels: 
support vector machines, regularization, optimization, and beyond.
Adaptive computation and machine learning. MIT Press, Cambridge, Mass.

## Examples
```julia
using JLD2, CairoMakie
mypath = joinpath(@__DIR__, "..", "data")
db = string(mypath, "\\", "cassav.jld2") 
@load db dat
pnames(dat)

X = dat.X 
y = dat.Y.y
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

gamma = .1 ; cost = 1000 ; epsilon = 1
fm = svmr(Xtrain, ytrain; kern = "krbf", 
        gamma = gamma, cost = cost, epsilon = epsilon) ;
pnames(fm)

res = predict(fm, Xtest)
res.pred
rmsep(res.pred, ytest)
f, ax = scatter(vec(res.pred), ytest)
abline!(ax, 0, 1)
f

# Example of fitting the function sinc(x)
# described in Rosipal & Trejo 2001 p. 105-106 
x = collect(-10:.2:10) 
x[x .== 0] .= 1e-5
n = length(x)
zy = sin.(abs.(x)) ./ abs.(x) 
y = zy + .2 * randn(n) 
fm = svmr(x, y; gamma = .1) ;
pred = predict(fm, x).pred 
f, ax = scatter(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "Fitted model")
axislegend("Method")
f
```
""" 
function svmr(X, y; kern = "krbf", 
    gamma = 1. / size(X, 2), degree = 3, coef0 = 0., cost = 1., 
    epsilon = .1)
    gamma = Float64(gamma) ; degree = Int64(degree) ; coef0  = Float64(coef0) ; 
    cost  = Float64(cost) ; epsilon = Float64(epsilon) ; 
    X = ensure_mat(X)
    y = vec(y)
    if kern == "krbf"
        fkern = LIBSVM.Kernel.RadialBasis
    elseif kern == "kpol"
        fkern = LIBSVM.Kernel.Polynomial
    elseif kern == "klin"
        fkern = LIBSVM.Kernel.Linear
    elseif kern == "ktanh"
        fkern = LIBSVM.Kernel.Sigmoid
    end
    nt = 0 
    fm = svmtrain(X', y;
        svmtype = EpsilonSVR, 
        kernel = fkern,
        gamma =  gamma,
        coef0 = coef0,
        degree = degree,
        cost = cost, epsilon = epsilon,
        nt = nt) 
    Svmr(fm)
end


"""
    predict(object::Svmr, X)
Compute y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
""" 
function predict(object::Svmr, X)
    X = ensure_mat(X)
    pred = svmpredict(object.fm, X')[1]
    n = length(pred)
    pred = reshape(pred, n, 1)
    (pred = pred,)
end
