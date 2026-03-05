struct ProtoClustPlsr
    fitm
    fitm_clust
    ycla::AbstractVector
end

function protoclustplsr(X, y; n_proto, metric = :eucl, nlvmax, scal = false, 
        k = 1, h = 1, criw = 3, squared = false, tolw = 1e-4)
    if metric == :eucl
        distance = Distances.Euclidean()
    elseif metric == :cos
        distance = Jchemo.CosDist()
    elseif metric == :was
        distance =  Jchemo.CorDist_b() # Jchemo.WasDist()
    end
    fitm_clust = kmeans(X', n_proto; 
        init = :kmpp,    # default
        maxiter = 5000, 
        display = :none,
        distance = distance,
        #distance = Distances.CosineDist(), 
        rng = Random.MersenneTwister(1234),  # set a constant seed
        ) 
    ycla = fitm_clust.assignments
    fitm = proto_ycla_plsr(X, y, ycla; metric, nlvmax, scal, k, h, criw) 
    ProtoClustPlsr(fitm, fitm_clust, ycla)
end

function Jchemo.predict(object::ProtoClustPlsr, X)
    predict_proto_ycla_plsr(object.fitm, X) 
end



