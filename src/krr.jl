"""
    krr(; kwargs...)
    krr(X, Y; kwargs...)
    krr(X, Y, weights::Weight; kwargs...)
    krr!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
Kernel ridge regression (KRR) implemented by SVD factorization.
* `X` : X-data (n, p).
* `Y` : Y-data (n, q).
* `weights` : Weights (n) of the observations. 
    Must be of type `Weight` (see e.g. function `mweight`).
Keyword arguments:
* `lb` : Ridge regularization parameter "lambda".
* `kern` : Type of kernel used to compute the Gram matrices.
    Possible values are: `:krbf`, `:kpol`. See respective 
    functions `krbf` and `kpol` for their keyword arguments.
* `scal` : Boolean. If `true`, each column of `X 
    is scaled by its uncorrected standard deviation.

KRR is also referred to as least squared SVM regression 
(LS-SVMR). The method is close to the particular case of 
SVM regression where there is no marge excluding the 
observations (epsilon coefficient set to zero). The difference 
is that a L2-norm optimization is done, instead of L1 in SVM.

## References 
Bennett, K.V., Embrechts, M.J., 2003. An optimization 
perspective on kernel partial least squares regression, 
in: Advances in Learning Theory: Methods, Models and 
Applications, NATO Science Series III: Computer & Systems 
Sciences. IOS Press Amsterdam, pp. 227-250.

Cawley, G.C., Talbot, N.L.C., 2002. Reduced Rank Kernel 
Ridge Regression. Neural Processing Letters 16, 293-302. 
https://doi.org/10.1023/A:1021798002258

Krell, M.M., 2018. Generalizing, Decoding, and Optimizing 
Support Vector Machine Classification. arXiv:1801.04929.

Saunders, C., Gammerman, A., Vovk, V., 1998. Ridge Regression 
Learning Algorithm in Dual Variables, in: In Proceedings of the 
15th International Conference on Machine Learning. Morgan 
Kaufitmann, pp. 515-521.

Suykens, J.A.K., Lukas, L., Vandewalle, J., 2000. Sparse 
approximation using least squares support vector machines. 2000 IEEE 
International Symposium on Circuits and Systems. Emerging Technologies 
for the 21st Century. Proceedings (IEEE Cat No.00CH36353).
https://doi.org/10.1109/ISCAS.2000.856439

Welling, M., n.d. Kernel ridge regression. Department of 
Computer Science, University of Toronto, Toronto, Canada. 
https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-Ridge.pdf

## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
path_jdat = dirname(dirname(pathof(JchemoData)))
db = joinpath(path_jdat, "data/cassav.jld2") 
@load db dat
@names dat
X = dat.X 
y = dat.Y.tbc
year = dat.Y.year
tab(year)
s = year .<= 2012
Xtrain = X[s, :]
ytrain = y[s]
Xtest = rmrow(X, s)
ytest = rmrow(y, s)

lb = 1e-3
kern = :krbf ; gamma = 1e-1
model = krr(; lb, kern, gamma) ;
fit!(model, Xtrain, ytrain)
@names model
@names model.fitm

coef(model)

res = predict(model, Xtest)
@head res.pred
@show rmsep(res.pred, ytest)
plotxy(res.pred, ytest; color = (:red, .5), bisect = true, xlabel = "Prediction", 
   ylabel = "Observed").f    

coef(model; lb = 1e-1)
res = predict(model, Xtest; lb = [.1 ; .01])
@head res.pred[1]
@head res.pred[2]

lb = 1e-3
kern = :kpol ; degree = 1
model = krr(; lb, kern, degree) 
fit!(model, Xtrain, ytrain)
res = predict(model, Xtest)
rmsep(res.pred, ytest)

####### Example of fitting the function sinc(x)
####### described in Rosipal & Trejo 2001 p. 105-106 
x = collect(-10:.2:10) 
x[x .== 0] .= 1e-5
n = length(x)
zy = sin.(abs.(x)) ./ abs.(x) 
y = zy + .2 * randn(n) 
lb = 1e-1
kern = :krbf ; gamma = 1 / 3
model = krr(; lb, kern, gamma) 
fit!(model, x, y)
pred = predict(model, x).pred 
f, ax = scatter(x, y) 
lines!(ax, x, zy, label = "True model")
lines!(ax, x, vec(pred), label = "Fitted model")
axislegend("Method")
f
```
""" 
krr(; kwargs...) = JchemoModel(krr, nothing, kwargs)

function krr(X, Y; kwargs...)
    Q = eltype(X[1, 1])
    weights = mweight(ones(Q, nro(X)))
    krr(X, Y, weights; kwargs...)
end

function krr(X, Y, weights::Weight; kwargs...)
    krr!(copy(ensure_mat(X)), copy(ensure_mat(Y)), weights; kwargs...)
end

function krr!(X::Matrix, Y::Union{Matrix, BitMatrix}, weights::Weight; kwargs...)
    par = recovkw(ParKrr, kwargs).par
    @assert in([:krbf ; :kpol])(par.kern) "Wrong value for argument 'kern'." 
    Q = eltype(X)
    isa(Y, BitMatrix) ? Y = convert.(Q, Y) : nothing
    p = nco(X)
    xscales = ones(Q, p)
    if par.scal 
        xscales .= colstd(X, weights)
        X = fscale(X, xscales)
    end
    ymeans = colmean(Y, weights)
    fkern = eval(Meta.parse(string("Jchemo.", par.kern)))
    K = fkern(X, X; kwargs...)
    sqrtw = sqrt.(weights.w)
    DKt = fweight(K', weights.w)
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- vtot .+ sum(fweight(DKt', weights.w))
    # Kd = D^(1/2) * Kc * D^(1/2) 
    #    = U * Delta^2 * U'    
    Kd = fweight(Kc, sqrtw) * Diagonal(sqrtw) 
    res = LinearAlgebra.svd(Kd)
    U = Matrix(res.V)
    sv = sqrt.(res.S)
    # UtDY = U' * D^(1/2) * Y
    UtDY = U' * fweight(Y, sqrtw)
    Krr(X, K, U, UtDY, sv, DKt, vtot, xscales, ymeans, weights, kwargs, par) 
end

"""
    coef(object::Krr; lb = nothing)
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
* `lb` : Ridge regularization parameter 
    "lambda".
""" 
function coef(object::Krr; lb = nothing)
    isnothing(lb) ? lb = object.par.lb : nothing
    lb = convert(eltype(object.X), lb)
    eig = object.sv.^2
    z = 1 ./ (eig .+ lb^2)
    A = object.U * (Diagonal(z) * object.UtDY)
    q = length(object.ymeans)
    int = reshape(object.ymeans, 1, q)
    tr = sum(eig .* z)
    (A = A, int = int, df = 1 + tr)
end

"""
    predict(object::Krr, X; lb = nothing)
Compute Y-predictions from a fitted model.
* `object` : The fitted model.
* `X` : X-data for which predictions are computed.
* `lb` : Regularization parameter, or collection of 
    regularization parameters, "lambda" to consider. 
""" 
function predict(object::Krr, X; lb = nothing)
    X = ensure_mat(X)
    isnothing(lb) ? lb = object.par.lb : nothing
    fkern = eval(Meta.parse(String(object.par.kern)))
    K = fkern(fscale(X, object.xscales), object.X; object.kwargs...)
    DKt = fweight(K', object.weights.w)
    vtot = sum(DKt, dims = 1)
    w = object.weights.w
    Kc = K .- vtot' .- object.vtot .+ sum(fweight(object.DKt', w))
    le_lb = length(lb)
    pred = list(Matrix{eltype(X)}, le_lb)
    @inbounds for i = 1:le_lb
        z = coef(object; lb = lb[i])
        pred[i] = z.int .+ Kc * (fweight(z.A, sqrt.(w)))
    end 
    le_lb == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end
