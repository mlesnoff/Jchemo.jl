struct Lda
    W::Array{Float64}  
    ct::Array{Float64}
    wprior::AbstractVector
    lev::AbstractVector
    ni::AbstractVector
end

"""
    lda(X, y; prior = "unif")
Linear discriminant analysis  (LDA).
* `X` : X-data.
* `y` : y-data (class membership).
* `prior` : Type of prior probabilities for class membership
    (`unif`: uniform; `prop`: proportional).
""" 
function lda(X, y; prior = "unif")
    X = ensure_mat(X)
    n = size(X, 1)
    z = aggstat(X, y; fun = mean)
    ct = z.res
    lev = z.lev
    nlev = length(lev)
    ni = z.ni
    if isequal(prior, "unif")
        wprior = ones(nlev) / nlev
    elseif isequal(prior, "prop")
        wprior = mweight(ni)
    end
    res = matW(X, y)
    res.W .= res.W * n / (n - nlev) # Unbiased estimate
    Lda(res.W, ct, wprior, lev, ni)
end

function predict(object::Lda, X)
    X = ensure_mat(X)
    m = size(X, 1)
    lev = object.lev
    nlev = length(lev) 
    ds = similar(X, m, nlev)
    for i = 1:nlev
        fm = dmnorm(; mu = object.ct[i, :], S = object.W) 
        ds[:, i] .= vec(predict(fm, X).pred)
    end
    A = object.wprior' .* ds
    v = sum(A, dims = 2)
    posterior = scale(A', v)' # This could be replaced by code similar as in scale! 
    z =  mapslices(argmax, posterior; dims = 2)  # if equal, argmax takes the first
    pred = reshape(replacebylev(z, object.lev), m, 1)
    (pred = pred, ds = ds, posterior = posterior)
end
    


