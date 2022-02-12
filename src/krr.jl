struct Krr
    X::Array{Float64}
    K::Array{Float64}
    U::Array{Float64}
    UtDY::Array{Float64}
    sv::Vector{Float64}
    D::Array{Float64}
    sqrtD::Array{Float64}
    DKt::Array{Float64}
    vtot::Array{Float64}
    lb::Float64
    ymeans::Vector{Float64}
    weights::Vector{Float64}
    kern
    dots
end

"""
    krr(X, Y, weights = ones(size(X, 1)); lb = .01, kern = "krbf", kwargs...)
Kernel ridge regression (KRR) implemented by SVD factorization.
* `X` : X-data.
* `Y` : Y-data.
* `weights` : Weights of the observations.
* `lb` : A value of the regularization parameter "lambda".
* 'kern' : Type of kernel used to compute the Gram matrices.
    Possible values are "krbf" of "kpol" (see respective functions `krbf` and `kpol`.
* `kwargs` : Named arguments to pass in the kernel function.

KRR is also referred to as least squared SVM regression (LS-SVMR).
The method is close to the particular case of SVM regression 
where there is novmarges excluding the observations (epsilon coefficient 
set to zero). The difference is that a L2-norm optimization is done, 
instead of L1 in SVM.

The kernel Gram matrices are internally centered. 

## References 

Bennett, K.P., Embrechts, M.J., 2003. An optimization perspective on kernel partial least squares regression, 
in: Advances in Learning Theory: Methods, Models and Applications, 
NATO Science Series III: Computer & Systems Sciences. IOS Press Amsterdam, pp. 227-250.

Cawley, G.C., Talbot, N.L.C., 2002. Reduced Rank Kernel Ridge Regression. 
Neural Processing Letters 16, 293-302. https://doi.org/10.1023/A:1021798002258

Krell, M.M., 2018. Generalizing, Decoding, and Optimizing Support Vector Machine Classification. 
arXiv:1801.04929.

Saunders, C., Gammerman, A., Vovk, V., 1998. Ridge Regression Learning Algorithm in Dual Variables, 
in: In Proceedings of the 15th International Conference on Machine Learning. Morgan Kaufmann, pp. 515â521.

Suykens, J.A.K., Lukas, L., Vandewalle, J., 2000. Sparse approximation using 
least squares support vector machines. 2000 IEEE International Symposium on Circuits and Systems. 
Emerging Technologies for the 21st Century. Proceedings (IEEE Cat No.00CH36353). https://doi.org/10.1109/ISCAS.2000.856439

Welling, M., n.d. Kernel ridge regression. Department of Computer Science, 
University of Toronto, Toronto, Canada. https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-Ridge.pdf

""" 
function krr(X, Y, weights = ones(size(X, 1)); lb = .01, kern = "krbf", kwargs...)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = mweight(weights)
    ymeans = colmean(Y, weights)   
    fkern = eval(Meta.parse(kern))    
    K = fkern(X, X; kwargs...)
    D = Diagonal(weights)    
    DKt = D * K'
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- vtot .+ sum(D * DKt')
    # Kd = D^(1/2) * Kc * D^(1/2) 
    #    = U * Delta^2 * U'    
    sqrtD = sqrt.(D)
    Kd = sqrtD * Kc * sqrtD
    res = LinearAlgebra.svd(Kd)
    U = res.V
    sv = sqrt.(res.S)
    # UtDY = U' * D^(1/2) * Y
    UtDY = U' * sqrtD * Y
    Krr(X, K, U, UtDY, sv, D, sqrtD, DKt, 
        vtot, lb, ymeans, weights, kern, kwargs)
end

"""
    coef(object::Krr; lb = nothing)
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
* `lb` : A value of the regularization parameter "lambda".
    If nothing, it is the parameter stored in the fitted model.
""" 
function coef(object::Krr; lb = nothing)
    isnothing(lb) ? lb = object.lb : nothing
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
* `object` : The maximal fitted model.
* `X` : X-data for which predictions are computed.
* `lb` : Regularization parameter, or collection of regularization parameters, "lambda" to consider. 
    If nothing, it is the parameter stored in the fitted model.
""" 
function predict(object::Krr, X; lb = nothing)
    isnothing(lb) ? lb = object.lb : nothing
    fkern = eval(Meta.parse(object.kern))
    K = fkern(X, object.X; object.dots...)
    DKt = object.D * K'
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- object.vtot .+ sum(object.D * object.DKt')
    le_lb = length(lb)
    pred = list(le_lb, Matrix{Float64})
    @inbounds for i = 1:le_lb
        z = coef(object; lb = lb[i])
        pred[i] = z.int .+ Kc * (object.sqrtD * z.A)
    end 
    le_lb == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end
