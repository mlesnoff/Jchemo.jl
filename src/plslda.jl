struct PlsProbDa
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

"""
    plslda(X, y, weights = ones(size(X, 1)); nlv)
LDA on PLS latent variables.
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations.
* `nlv` : Nb. latent variables (LVs) to compute.

The training variable y (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy variable (0/1). 
Then, a PLS2 is implemented on the X-data and Ydummy, 
returning `nlv` latent variables (LVs).
Finally, a LDA is run on these LVs and `y`. 
""" 
function plslda(X, y, weights = ones(size(X, 1)); nlv, prior = "unif")
    z = dummy(y)
    fm_pls = plskern(X, z.Y, weights; nlv = nlv)
    fm_da = list(nlv)
    for i = 1:nlv
        fm_da[i] = lda(fm_pls.T[:, 1:i], y; prior = prior)
    end
    fm = (fm_pls = fm_pls, fm_da = fm_da)
    PlsProbDa(fm, z.lev, z.ni)
end

function predict(object::PlsProbDa, X; nlv = nothing)
    X = ensure_mat(X)
    m = size(X, 1)
    a = size(object.fm.fm_pls.T, 2)
    isnothing(nlv) ? nlv = a : nlv = (max(minimum(nlv), 0):min(maximum(nlv), a))
    le_nlv = length(nlv)
    pred = list(le_nlv)
    posterior = list(le_nlv)
    @inbounds for i = 1:le_nlv
        znlv = nlv[i]
        T = transform(object.fm.fm_pls, X, nlv = znlv)
        zres = predict(object.fm.fm_da[znlv], T)
        z =  mapslices(argmax, zres.posterior; dims = 2) 
        pred[i] = reshape(replacebylev(z, object.lev), m, 1)
        posterior[i] = zres.posterior
    end 
    if le_nlv == 1
        pred = pred[1]
        posterior = posterior[1]
    end
    (pred = pred, posterior = posterior)
end

"""
    plsqda(X, y, weights = ones(size(X, 1)); nlv)
QDA on PLS latent variables.
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations.
* `nlv` : Nb. latent variables (LVs) to compute.

The training variable y (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy variable (0/1). 
Then, a PLS2 is implemented on the X-data and Ydummy, 
returning `nlv` latent variables (LVs).
Finally, a QDA is run on these LVs and `y`. 
""" 
function plsqda(X, y, weights = ones(size(X, 1)); nlv, prior = "unif")
    z = dummy(y)
    fm_pls = plskern(X, z.Y, weights; nlv = nlv)
    fm_da = list(nlv)
    for i = 1:nlv
        fm_da[i] = qda(fm_pls.T[:, 1:i], y; prior = prior)
    end
    fm = (fm_pls = fm_pls, fm_da = fm_da)
    PlsProbDa(fm, z.lev, z.ni)
end



