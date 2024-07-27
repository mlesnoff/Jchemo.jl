function defpar(fun::Function)

    in((nipals, nipalsmiss)).(fun) ? dump(Jchemo.ParNipals()) : nothing

    in((pcasvd, pcaeigen, pcaeigenk, pcasph)).(fun) ? dump(Jchemo.ParPca()) : nothing
    in((pcanipals, pcanipalsmiss)).(fun) ? dump(Jchemo.ParPcanipals()) : nothing
    in((pcapp,)).(fun) ? dump(Jchemo.ParPcapp()) : nothing
    in((pcaout,)).(fun) ? dump(Jchemo.ParOut()) : nothing

    in((fda, fdasvd)).(fun) ? dump(Jchemo.ParFda()) : nothing

    in((kpca,)).(fun) ? dump(Jchemo.ParKpca()) : nothing

    in((blockscal,)).(fun) ? dump(Jchemo.ParBlock()) : nothing
    in((plscan, plstuck)).(fun) ? dump(Jchemo.ParPls2bl()) : nothing
    in((mbpca, comdim, )).(fun) ? dump(Jchemo.ParMbpca()) : nothing
    in((cca,)).(fun) ? dump(Jchemo.ParCca()) : nothing
    in((ccawold,)).(fun) ? dump(Jchemo.ParCcawold()) : nothing

    in((mlr, mlrpinv, mlrvec)).(fun) ? dump(Jchemo.ParMlr()) : nothing

    in((pcr,)).(fun) ? dump(Jchemo.ParPcr()) : nothing

    in((plskern, plsnipals, plswold, plsrosa, plssimp, plsravg, plsravg_unif)).(fun) ? dump(Jchemo.ParPlsr()) : nothing
    in((cglsr, aicplsr)).(fun) ? dump(Jchemo.ParCglsr()) : nothing
    in((plsrout,)).(fun) ? dump(Jchemo.ParPlsrout()) : nothing

    in((krr,)).(fun) ? dump(Jchemo.ParKrr()) : nothing
    in((kplsr, dkplsr)).(fun) ? dump(Jchemo.ParKplsr()) : nothing

    in((knnr, lwmlr)).(fun) ? dump(Jchemo.ParKnn()) : nothing
    in((lwplsr, lwplsravg)).(fun) ? dump(Jchemo.ParLwplsr()) : nothing

    in((mbplsr, mbplswest)).(fun) ? dump(Jchemo.ParMbplsr()) : nothing

    in((dmnorm, dmnormlog)).(fun) ? dump(Jchemo.ParDmnorm()) : nothing
    in((dmkern,)).(fun) ? dump(Jchemo.ParDmkern()) : nothing

    in((lda,)).(fun) ? dump(Jchemo.ParLda()) : nothing
    in((kdeda,)).(fun) ? dump(Jchemo.ParKdeda()) : nothing
    
    in((mlrda,)).(fun) ? dump(Jchemo.ParMlrda()) : nothing
    in((plsrda, plslda)).(fun) ? dump(Jchemo.ParPlsda()) : nothing
    in((plsqda,)).(fun) ? dump(Jchemo.ParPlsqda()) : nothing
    in((plskdeda,)).(fun) ? dump(Jchemo.ParPlskdeda()) : nothing

    in((krrda,)).(fun) ? dump(Jchemo.ParKrrda()) : nothing
    in((kplsrda, kplslda, dkplsrda, dkplslda)).(fun) ? dump(Jchemo.ParKplsda()) : nothing
    in((kplsqda, dkplsqda)).(fun) ? dump(Jchemo.ParKplsqda()) : nothing
    in((kplskdeda, dkplskdeda)).(fun) ? dump(Jchemo.ParKplskdeda()) : nothing

    in((knnda, lwmlrda)).(fun) ? dump(Jchemo.ParKnn()) : nothing
    in((lwplsrda, lwplslda)).(fun) ? dump(Jchemo.ParLwplsda()) : nothing
    in((lwplsqda,)).(fun) ? dump(Jchemo.ParLwplsqda()) : nothing

    in((mbplsrda, mbplslda)).(fun) ? dump(Jchemo.ParMbplsda()) : nothing
    in((mbplsqda,)).(fun) ? dump(Jchemo.ParMbplsqda()) : nothing
    in((mbplskdeda,)).(fun) ? dump(Jchemo.ParMbplskdeda()) : nothing

    in((outeucl, outstah,)).(fun) ? dump(Jchemo.ParOut()) : nothing
    in((occsd, occod, occsdod)).(fun) ? dump(Jchemo.ParOcc()) : nothing
    in((occstah,)).(fun) ? dump(Jchemo.ParOut()) : nothing

end

