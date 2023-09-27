function mtestsys(Y::DataFrame, id = 1:nro(Y); ntest)
    nam = names(Y)
    p = length(nam)
    idtest = list(p, Vector)
    idtrain = list(p, Vector)  
    length(ntest) == 1 ? ntest = repeat([ntest], p) : nothing
    for i = 1:p
        y = Y[:, nam[i]]
        s_all = findall(ismissing.(y) .== 0)      
        ntot = length(s_all)
        s_test = sampsys(y; k = ntest[i])         
        s_train = (1:ntot)[in(s_test).(1:ntot) .== 0]
        idtest[i] = sort(id[s_all[s_test]])      
        idtrain[i] = sort(id[s_all[s_train]])
    end
    (test = idtest, train = idtrain, nam)
end

