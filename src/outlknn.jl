"""
    outlknn(X; metric = :eucl, k, algo = sum, scal::Bool = false)
    outlknn!(X::Matrix; metric = :eucl, k, algo = sum, scal::Bool = false)
Compute a local kNN distance-based outlierness.
* `X` : X-data (n, p).
Keyword arguments:
* `metric` : Metric used to compute the dustances. See function `getknn`.
* `k` : Nb. nearest neighbors to consider.
* `algo` : Function summarizing the distances to the neighbors.
* `scal` : Boolean. If `true`, each column of `X` is scaled before computing the outlierness.

The idea is to compare the KNN-outlierness of the observation to the KNN-outlierness of its neighbors, giving a local 
measure of outlierness. For each observation (row of `X`), the outlierness is defined as folloxs:

* A summary (e.g. by sum) of the distances between the observation and its `k` nearest neighbors
    is computed, say out1.
* The same summary is computed for each of the `k` nearest neighbors of the observation, and the average of 
    the `k` returned values is computed, say out2.
* The outlierness of the observation is finally defined as the ratio out1 / out2.

The approach can be seen as a simplification of the local outlier factor (LOF) method (Breunig et al. 2000),
such as the Simplified-LOF method (Schubert et al 2014 p.206, Campos et al. 2016 p.896) where local density 
is estimated by the inverse of the k-distance.

## References
Campos, G.O., Zimek, A., Sander, J., Campello, R.J.G.B., Micenková, B., Schubert, E., Assent, I., Houle, M.E., 2016. 
On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study. Data Min Knowl 
Disc 30, 891–927. https://doi.org/10.1007/s10618-015-0444-8

Schubert, E., Zimek, A., Kriegel, H.-P., 2014. Local outlier detection reconsidered: a generalized view on 
locality with applications to spatial, video, and network outlier detection. Data Min Knowl Disc 28, 190–237. 
https://doi.org/10.1007/s10618-012-0300-z


## Examples
```julia
using Jchemo, JchemoData, JLD2, CairoMakie
mypath = dirname(dirname(pathof(JchemoData)))
db = joinpath(mypath, "data", "octane.jld2")
@load db dat
X = dat.X
wlst = names(X)
wl = parse.(Float64, wlst)
n, p = size(X)
## Six of the samples (25, 26, and 36-39) contain added alcohol.
s = [25; 26; 36:39]
typ = zeros(Int, n)
typ[s] .= 1
#plotsp(X, wl; xlabel = "Wavelength (nm)", ylabel = "Absorbance").f

metric = :eucl ; k = 15 ; algo = maximum
#algo = :maximum
res = outlknn(X; metric, k, algo) ;
@names res
f, ax = plotxy(1:n, res.d, typ, xlabel = "Obs. index", ylabel = "Outlierness")
text!(ax, 1:n, res.d; text = string.(1:n), fontsize = 10)
f

nlv = 3
model = pcasph(; nlv)
fit!(model, X)
T = model.fitm.T
metric = :eucl 
k = 15
res = outlknn(T; metric, k, scal)
plotxy(1:n, res.d, typ, xlabel = "Obs. index", ylabel = "Outlierness").f
```
""" 
function outlknn(X; metric = :eucl, k, algo = sum, scal::Bool = false)
    outlknn!(copy(ensure_mat(X)); k, metric, algo, scal)
end

function outlknn!(X::Matrix; metric = :eucl, k, algo = sum, scal::Bool = false)
    Q = eltype(X)
    n, p = size(X)
    xscales = ones(Q, p)
    if scal
        xscales .= colstd(X)
        fscale!(X, xscales)
    end
    k > n - 1 ? k = n - 1 : nothing
    res = getknn(X, X; k = k + 1, metric)
    d = zeros(n)
    nn = zeros(Int, k)
    @inbounds for i in eachindex(d)
        d[i] = algo(res.d[i][2:end])
        nn .= res.ind[i][2:end]
        res_nn = getknn(X, vrow(X, nn); k = k + 1, metric)
        d_nn = zeros(Q, k) 
        for j in eachindex(nn)
            d_nn[j] = algo(res_nn.d[j][2:end])
        end
        d[i] /= mean(d_nn)
    end
    (d = d, xscales)
end



