struct ProtoClustPlsr
    fitm::ProtoYclaPlsr
    fitm_clust::Clustering.KmeansResult
    ycla::AbstractVector
end

## Not exported
function protoclustplsr(X, y; metric = :eucl, nproto, nlv, kavg = 1, h = 1, criw = 3, squared = false, 
        tolw = 1e-4, scal = false)
    if metric == :eucl
        distance = Distances.Euclidean()
    elseif metric == :cos
        distance = Jchemo.CosDist()
    elseif metric == :sam
        distance = Jchemo.SamDist()
    elseif metric == :cor 
        distance = Jchemo.CorDist()
    #elseif metric == :was
    #    distance =  Jchemo.CorDist_b() # Jchemo.WasDist()
    end
    fitm_clust = kmeans(X', nproto; 
        init = :kmpp,    # default
        maxiter = 5000, 
        display = :none,
        distance = distance,
        rng = Random.MersenneTwister(1234),  # set a constant seed
        ) 
    ycla = fitm_clust.assignments
    fitm = Jchemo.protoyclaplsr(X, y, ycla; metric, nlv, kavg, h, criw, squared, tolw, scal) 
    ProtoClustPlsr(fitm, fitm_clust, ycla)
end

function predict(object::ProtoClustPlsr, X)
    predict(object.fitm, X) 
end



