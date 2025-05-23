"""
    outknn(X; metric = :eucl, k, algo = sum, scal::Bool = false)
    outknn!(X::Matrix; metric = :eucl, k, algo = sum, scal::Bool = false)
Compute a kNN distance-based outlierness.
* `X` : X-data (n, p).
Keyword arguments:
* `metric` : Metric used to compute the distances. See function `getknn`.
* `k` : Nb. nearest neighbors to consider.
* `algo` : Function summarizing the `k` distances to the neighbors.
* `scal` : Boolean. If `true`, each column of `X` is scaled before computing the outlierness.

For each observation (row of `X`), the outlierness is defined by a summary (e.g. by sum or maximum) of the distances 
between the observation and its `k` nearest neighbors. 

## References
Angiulli, F., Pizzuti, C., 2005. Outlier mining in large high-dimensional data sets. IEEE Transactions on Knowledge 
and Data Engineering 17, 203–215. https://doi.org/10.1109/TKDE.2005.31

Angiulli, F., Basta, S., Pizzuti, C., 2006. Distance-based detection and prediction of outliers. IEEE Transactions 
on Knowledge and Data Engineering 18, 145–160. https://doi.org/10.1109/TKDE.2006.29

Campos, G.O., Zimek, A., Sander, J., Campello, R.J.G.B., Micenková, B., Schubert, E., Assent, I., Houle, M.E., 2016. 
On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study. Data Min Knowl 
Disc 30, 891–927. https://doi.org/10.1007/s10618-015-0444-8

Ramaswamy, S., Rastogi, R., Shim, K., 2000. Efficient algorithms for mining outliers from large data sets, 
in: Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data, SIGMOD ’00. 
Association for Computing Machinery, New York, NY, USA, pp. 427–438. https://doi.org/10.1145/342009.335437

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

metric = :eucl ; k = 15 ; algo = sum
#algo = maximum
res = outknn(X; metric, k, algo) ;
@names res
f, ax = plotxy(1:n, res.d, typ, xlabel = "Obs. index", ylabel = "Outlierness")
text!(ax, 1:n, res.d; text = string.(1:n), fontsize = 10)
f

## With a preliminary PCA
nlv = 3
model = pcasph(; nlv)
fit!(model, X)
T = model.fitm.T
metric = :eucl 
k = 15
res = outknn(T; metric, k, scal = true)
plotxy(1:n, res.d, typ, xlabel = "Obs. index", ylabel = "Outlierness").f
```
""" 
function outknn(X; metric = :eucl, k, algo = sum, scal::Bool = false)
    outknn!(copy(ensure_mat(X)); k, metric, algo, scal)
end

function outknn!(X::Matrix; metric = :eucl, k, algo = sum, scal::Bool = false)
    Q = eltype(X)
    n, p = size(X)
    xscales = ones(Q, p)
    if scal
        xscales .= colstd(X)
        fscale!(X, xscales)
    end
    k > n - 1 ? k = n - 1 : nothing
    res = getknn(X, X; k = k + 1, metric)
    d = similar(X, n)
    @inbounds for i in eachindex(d)
        d[i] = algo(res.d[i][2:end])
    end
    (d = d, xscales)
end



