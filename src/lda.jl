struct Lda1
    W::Array{Float64}  
    ct::Array{Float64}
    wprior::AbstractVector
    lev::AbstractVector
    ni::AbstractVector
end

"""
    lda(X, y, weights = ones(size(X, 1)))
Discrimination (DA) based on linear regression (LMR).
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations.

The training variable y (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy variable (0/1). 
Then, a linear regression model (LMR) is run on the X-data and Ydummy, 
returning predictions of the dummy variables. 
For a given observation, the final prediction is the class corresponding 
to the dummy variable for which the prediction is the highest.

By default, a softmax transformation is applied to the 
predicted Ydummy. This can be modified by argument `softmax`:
`predict(fm, Xtest; softmax = false)`.
""" 
function lda(X, y; prior = "unif")
    X = ensure_mat(X)
    z = aggstat(X, y; fun = mean)
    lev = z.lev
    nlev = length(lev)
    ni = z.ni
    isequal(prior, "unif") ? wprior = ones(nlev) / nlev : nothing
    isequal(prior, "prop") ? wprior = mweights(ni) : nothing
    res = matW(X, y)
    res.W .= res.W * n / (n - nlev)
    Lda1(res.W, z.ct, wprior, lev, ni)
end

function predict(object::Lda1, X; softmax = true)
    X = ensure_mat(X)
    m = size(X, m)
    lev = object.lev
    nlev = length(lev) 
    ds = similar(X, m, nlev)
    for i = 1:nlev
        fm = dmnorm(X; mu = object.ct[i, :], sigma = object.W) 
        ds[:, i] = vec(predict(fm, X).pred)
    end
    A = object.prior' .* ds
    v = sum(A, dims = 2)
    posterior = scale(A', v)' # This could be replaced by code similar as in scale! 
    z =  mapslices(argmax, posterior; dims = 2) 
    pred = reshape(replacebylev(z, object.lev), m, 1)
    list(pred = pred, ds = ds, posterior = posterior)
end
    


