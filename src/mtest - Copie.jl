
function mtest_old(Y::DataFrame, id = 1:nro(Y); ntest, 
        rep = 1)
    nam = names(Y)
    p = length(nam)
    idtrain = list(p, Vector)  
    idtest = list(p, Vector)
    length(ntest) == 1 ? ntest = repeat([ntest], p) : nothing
    for i = 1:p
        y = Y[:, nam[i]]
        s_all = findall(ismissing.(y) .== 0)      
        ntot = length(s_all)
        zidtrain = list(rep, Vector)
        zidtest = list(rep, Vector)
        for j = 1:rep 
            ntrain = ntot - ntest[i]
            res = samprand(ntot; k = ntrain)
            zidtest[j] = sort(id[s_all[res.test]])      
            zidtrain[j] = sort(id[s_all[res.train]])
        end
        idtest[i] = zidtest
        idtrain[i] = zidtrain
    end
    (train = idtrain, test = idtest, nam)
end

