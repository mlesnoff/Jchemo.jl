struct Qda
    Wi::AbstractVector  
    ct::Array{Float64}
    wprior::AbstractVector
    lev::AbstractVector
    ni::AbstractVector
end

"""
    qda(X, y; prior = "unif")
Quadratic discriminant analysis  (QDA).
* `X` : X-data.
* `y` : Univariate class membership.
* `prior` : Type of prior probabilities for class membership
    (`unif`: uniform; `prop`: proportional).
""" 
function qda(X, y; prior = "unif")
    X = ensure_mat(X)
    z = aggstat(X, y; fun = mean)
    ct = z.res
    lev = z.lev
    nlev = length(lev)
    ni = z.ni
    if isequal(prior, "unif")
        wprior = ones(nlev) / nlev
    elseif isequal(prior, "prop")
        wprior = mweights(ni)
    end
    res = matW(X, y)
    Qda(res.Wi, ct, wprior, lev, ni)
end

function predict(object::Qda, X)
    X = ensure_mat(X)
    m = size(X, 1)
    lev = object.lev
    nlev = length(lev) 
    ds = similar(X, m, nlev)
    ni = object.ni
    for i = 1:nlev
        if ni[i] == 1
            S = object.Wi[i] 
        else
            S = object.Wi[i] * ni[i] / (ni[i] - 1)
        end
        fm = dmnorm(; mu = object.ct[i, :], S = S) 
        ds[:, i] .= vec(predict(fm, X).pred)
    end
    A = object.wprior' .* ds
    v = sum(A, dims = 2)
    posterior = scale(A', v)' # This could be replaced by code similar as in scale! 
    z =  mapslices(argmax, posterior; dims = 2) 
    pred = reshape(replacebylev(z, object.lev), m, 1)
    (pred = pred, ds = ds, posterior = posterior)
end
    


