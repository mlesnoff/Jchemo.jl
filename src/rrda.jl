struct RrDa
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

struct KrrDa
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

"""
    rrda(X, y, weights = ones(size(X, 1)); lb)
Discrimination (DA) based on ridge regression (RR).
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations.
* `lb` : A value of the regularization parameter "lambda".

The training variable y (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy variable (0/1). 
Then, a RR is implemented on the X-data and each column of Ydummy,
returning predictions of the dummy variables (= object `posterior` returned by 
function `predict`). 
These predictions can be considered as unbounded (i.e. eventuall outside of [0, 1]) 
estimates of the class membership probabilities.
For a given observation, the final prediction is the class corresponding 
to the dummy variable for which the prediction is the highest.
""" 
function rrda(X, y, weights = ones(size(X, 1)); lb)
    z = dummy(y)
    fm = rr(X, z.Y, weights; lb = lb)
    RrDa(fm, z.lev, z.ni)
end

function predict(object::Union{RrDa, KrrDa}, X; lb = nothing)
    X = ensure_mat(X)
    m = size(X, 1)
    isnothing(lb) ? lb = object.fm.lb : nothing
    le_lb = length(lb)
    pred = list(le_lb)
    posterior = list(le_lb)
    @inbounds for i = 1:le_lb
        zp = predict(object.fm, X; lb = lb[i]).pred
        z =  mapslices(argmax, zp; dims = 2) 
        pred[i] = reshape(replacebylev(z, object.lev), m, 1)
        posterior[i] = zp
    end 
    if le_lb == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end
    
"""
    krrda(X, y, weights = ones(size(X, 1)); lb)
Discrimination (DA) based on kernel ridge regression (KRR).
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations.
* `lb` : A value of the regularization parameter "lambda".
* 'kern' : Type of kernel used to compute the Gram matrices.
    Possible values are "krbf" of "kpol" (see respective functions `krbf` and `kpol`.
* `kwargs` : Named arguments to pass in the kernel function.

The training variable y (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy variable (0/1). 
Then, a KRR is implemented on the X-data and each column of Ydummy,
returning predictions of the dummy variables (= object `posterior` returned by 
function `predict`). 
These predictions can be considered as unbounded (i.e. eventuall outside of [0, 1]) 
estimates of the class membership probabilities.
For a given observation, the final prediction is the class corresponding 
to the dummy variable for which the prediction is the highest.
""" 
function krrda(X, y, weights = ones(size(X, 1)); lb, kern = "krbf", kwargs...)
    z = dummy(y)
    fm = krr(X, z.Y, weights; lb = lb, kern = kern, kwargs...)
    KrrDa(fm, z.lev, z.ni)
end


