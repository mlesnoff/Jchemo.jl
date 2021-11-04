struct CplsrAgg1
    fm
    fm_da::PlsrDa
    lev
    ni
end

function cplsr_agg(X, Y, cla = nothing; ncla = nothing, nlv_da, nlv)
    X = ensure_mat(X) 
    Y = ensure_mat(Y)
    if isnothing(cla)
        zfm = Clustering.kmeans(X', ncla; maxiter = 500, display = :none)
        cla = zfm.assignments
    end
    z = tab(cla)
    lev = z.keys
    nlev = length(lev)
    ni = collect(values(z))
    fm_da = plsrda(X, cla; nlv = nlv_da)
    fm = list(nlev)
    @inbounds for i = 1:nlev
        s = findall(cla .== lev[i])
        z = eval(Meta.parse(nlv))
        zmin = minimum(z)
        zmax = maximum(z)
        ni[i] <= zmin ? zmin = ni[i] - 1 : nothing
        ni[i] <= zmax ? zmax = ni[i] - 1 : nothing
        znlv = string(zmin:zmax)
        fm[i] = plsr_agg(X[s, :], Y[s, :]; nlv = znlv)
    end
    CplsrAgg1(fm, fm_da, lev, ni)
end

function predict(object::CplsrAgg1, X)
    X = ensure_mat(X)
    m = size(X, 1)
    nlev = length(object.lev)
    zp = predict(object.fm_da, X).posterior
    zp .= (mapreduce(i -> Float64.(zp[i, :] .== maximum(zp[i, :])), hcat, 1:m)')
    #zp = (mapreduce(i -> Jchemo.mweights(exp.(zp[i, :])), hcat, 1:m))'
    #zp .= (mapreduce(i -> 1 ./ (1 .+ exp.(-zp[i, :])), hcat, 1:m)')
    #zp .= (mapreduce(i -> zp[i, :] / sum(zp[i, :]), hcat, 1:m))'
    acc = zp[:, 1] .* predict(object.fm[1], X).pred
    @inbounds for i = 2:nlev
        if object.ni[i] >= 30
            acc .+= zp[:, i] .* predict(object.fm[i], X).pred
        end
    end
    (pred = acc, posterior = zp)
end


