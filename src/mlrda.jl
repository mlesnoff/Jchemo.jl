struct Mlrda
    fm  
    lev::AbstractVector
    ni::AbstractVector
end

"""
    mlrda(X, y, weights = ones(size(X, 1)))
Discrimination based on linear regression (LMR-DA).
* `X` : X-data.
* `y` : Univariate class membership.
* `weights` : Weights of the observations.

The training variable y (univariate class membership) is transformed
to a dummy table (Ydummy) containing nlev columns, where nlev is the number 
of classes present in `y`. Each column of Ydummy is a dummy variable (0/1). 
Then, a linear regression model (LMR) is run on the X-data and and each column 
of Ydummy, returning predictions of the dummy variables. 
For a given observation, the final prediction is the class corresponding 
to the dummy variable for which the prediction is the highest.

By default, a softmax transformation is applied to the 
predicted Ydummy. This can be modified by argument `softmax`:
`predict(fm, Xtest; softmax = false)`.
""" 
function mlrda(X, y, weights = ones(size(X, 1)))
    z = dummy(y)
    fm = lmr(X, z.Y, weights)
    Mlrda(fm, z.lev, z.ni)
end

function predict(object::Mlrda, X; softmax = true)
    posterior = predict(object.fm, X).pred
    m = size(posterior, 1)
    if softmax
        @inbounds for i = 1:m
            posterior[i, :] .= mweights(exp.(posterior[i, :]))
        end
    end
    z =  mapslices(argmax, posterior; dims = 2) 
    pred = reshape(replacebylev(z, object.lev), m, 1)
    (pred = pred, posterior = posterior)
end
    


