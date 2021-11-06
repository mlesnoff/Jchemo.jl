struct Mlrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

"""
    mlrda(X, y, weights = ones(size(X, 1)))
Discrimination based on multple linear regression (MLR-DA).
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations.

The training variable y (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy variable (0/1). 
Then, a multiple linear regression (MLR) is run between the X-data and and each column 
of Ydummy, returning predictions of the dummy variables (object `posterior` 
retunerd by fuction `predict`).  
These predictions can be  considered as unbounded (i.e. eventuall outside of [0, 1]) 
estimates of the class memberships.
For a given observation, the final class prediction is the class corresponding 
to the dummy variable for which the prediction is the highest.
""" 
function mlrda(X, y, weights = ones(size(X, 1)))
    z = dummy(y)
    fm = mlr(X, z.Y, weights)
    Mlrda(fm, z.lev, z.ni)
end

function predict(object::Mlrda, X)
    X = ensure_mat(X)
    m = size(X, 1)
    zp = predict(object.fm, X).pred
    z =  mapslices(argmax, zp; dims = 2) 
    pred = reshape(replacebylev(z, object.lev), m, 1)
    (pred = pred, posterior = zp)
end
    


