predictcv = function(model, X, Y; segm, score)
    X = ensure_mat(X)
    Y = ensure_mat(Y)
    q = nco(Y)
    Q = eltype(Y)
    rep = length(segm)
    K = length(segm[1]) 
    matpred = Vector{Matrix{Q}}(undef, rep * K)
    maty = Vector{Matrix{Q}}(undef, rep * K)
    k = 1
    scor = repeat([0], 1, q)
    for i = 1:rep
        listsegm = segm[i]
        for j = 1:K
            s = listsegm[j]
            m = length(s)
            dat = hcat(repeat([i], m), repeat([j], m))           
            fit!(model, rmrow(X, s), rmrow(Y, s))
            pred = predict(model, vrow(X, s)).pred
            matpred[k] = hcat(dat, pred)
            maty[k] = hcat(dat, vrow(Y, s))
            scor = scor + score(pred, vrow(Y, s))
            k = k + 1
        end
    end
    nam = vcat([:rep, :segm], Symbol.(string.("y", 1:q)))
    typ = vcat([Int; Int], repeat([Q], q))
    mat = DataFrame(reduce(vcat, matpred), nam)
    matpred = convertdf(mat; typ)   
    mat = DataFrame(reduce(vcat, maty), nam)
    maty = convertdf(mat; typ)   
    scor = scor / (rep * K)
    (matpred = matpred, maty, scor)
end





