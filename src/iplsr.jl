# To do: Add a structure a make a surchage of 'plot' of Makie 

"""
    iplsr(Xtrain, Ytrain, X, Y; nint = 5, score = rmsep, nlv)
Interval PLSR (iPLS) (Nørgaard et al. 2000)
* `Xtrain` : Training X-data.
* `Ytrain` : Training Y-data.
* `X` : Validation X-data.
* `Y` : Validation Y-data.
* `nint` : Nb. intervals. 
* `score` : Function computing the prediction score (= error rate; e.g. msep).
* `nlv` : Nb. latent variables (LVs) in the PLSR models.

The range `1:p` (where `p` is the nb. of columns of `Xtrain`) is segmented
to `nint` column-intervals of (when possible) equal size. Then, the validation 
score is computed for each of the `nint` PLSR models and compared to the 
one of the overal `Xtrain`.

## References

- Nørgaard, L., Saudland, A., Wagner, J., Nielsen, J.P., Munck, L., 
    Engelsen, S.B., 2000. Interval Partial Least-Squares Regression (iPLS): 
    A Comparative Chemometric Study with an Example from Near-Infrared 
    Spectroscopy. Appl Spectrosc 54, 413–419. https://doi.org/10.1366/0003702001949500
"""
function iplsr(Xtrain, Ytrain, X, Y; 
        nint = 5, score = rmsep, nlv)
    Y = ensure_mat(Y) 
    p = size(Xtrain, 2)
    q = size(Y, 2)
    z = collect(round.(range(1, p + 1, length = nint + 1)))
    int = Int64.([z[1:nint] z[2:(nint + 1)] .- 1])
    pred = similar(Y)
    res = list(nint, Matrix{Float64})
    @inbounds for i = 1:nint
        u = int[i, 1]:int[i, 2]
        fm = plskern(vcol(Xtrain, u), Ytrain; nlv = nlv)
        pred .= predict(fm, vcol(X, u)).pred
        res[i] = score(pred, Y)
    end
    res = reduce(vcat, res)
    fm = plskern(Xtrain, Ytrain; nlv = nlv)
    pred = predict(fm, X).pred
    res0 = score(pred, Y)
    dat = DataFrame(int, [:lo, :up])
    namy = map(string, repeat(["y"], q), 1:q)
    res = DataFrame(res, Symbol.(namy))
    res = hcat(dat, res)
    res0 = DataFrame(res0, Symbol.(namy))
    (res = res, res0 = res0, int = int)
end



