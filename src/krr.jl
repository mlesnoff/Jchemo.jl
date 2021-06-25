struct KrrSvd1
    X::Array{Float64}
    K::Array{Float64}
    U::Array{Float64}
    UtDY::Array{Float64}
    sv::Vector{Float64}
    lb::Float64
    ymeans::Vector{Float64}
    weights::Vector{Float64}
    kern
    dots
end

"""
    krrsvd(X, Y, weights = ones(size(X, 1)); lb = .01 , kern = "krbf", kwargs...)
Kernel ridge regression (KRR) implemented by SVD factorization.
* `X` : matrix (n, p), or vector (n,).
* `Y` : matrix (n, q), or vector (n,).
* `weights` : vector (n,).
* `lb` : A value of the regularization parameter "lambda".

KRR is also referred to as least squared SVM regression (LS-SVMR).

The kernel Gram matrices are internally centered. 

The in-place version modifies `X` and `Y`. 

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
function krrsvd(X, Y, weights = ones(size(X, 1)); lb = .01, kern = "krbf", kwargs...)
    fkern = eval(Meta.parse(kern))
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    weights = mweights(weights)
    ymeans = colmeans(Y, weights)   
    K = fkern(X, X, kwargs...)
    D = Diagonal(weights)    
    DKt = D * K'
    vtot = sum(DKt, dims = 1)
    Kc = K .- vtot' .- vtot .+ sum(D * DKt')
    ## Kd = D^(1/2) * Kc * D^(1/2) 
    ##    = U * Delta^2 * U'    
    sqrtD = sqrt.(D)
    Kd = sqrtD * Kc * sqrtD
    res = LinearAlgebra.svd!(Kd)
    U = res.V
    sv = sqrt.(res.S)
    ## UtDY = U' * D^(1/2) * Y
    UtDY = U' * sqrtD * Y
    dots = nothing #kwargs...
    KrrSvd1(X, K, U, UtDY, sv, lb, ymeans, weights, kern, dots)
end

"""
    coef(object::KrrSvd1; lb = nothing)
Compute the b-coefficients of a fitted model.
* `object` : The fitted model.
* `lb` : A value of the regularization parameter "lambda".
If nothing, it is the parameter stored in the fitted model.
""" 
function coef(object::KrrSvd1; lb = nothing)
    isnothing(lb) ? lb = object.lb : nothing
    eig = object.sv.^2
    z = 1 ./ (eig .+ lb^2)
    A = object.U * (Diagonal(z) * object.UtDY)
    int = Matrix(object.ymeans')
    tr = sum(eig .* z)
    (int = int, A = A, df = 1 + tr)
end








"""
    predict(object::RrSvd, X; lb = nothing)
Compute Y-predictions from a fitted model.
* `object` : The maximal fitted model.
* `X` : Matrix (m, p) for which predictions are computed.
* `lb` : Regularization parameter, or collection of regularization parameters, "lambda" to consider. 
If nothing, it is the parameter stored in the fitted model.
""" 
function predict(object::KrrSvd1, X; lb = nothing)
    isnothing(lb) ? lb = object.lb : nothing
    le_lb = length(lb)
    pred = list(le_lb)
    @inbounds for i = 1:le_lb
        z = coef(object; lb = lb[i])
        pred[i] = z.int .+ X * z.B
    end 
    le_lb == 1 ? pred = pred[1] : nothing
    (pred = pred,)
end
