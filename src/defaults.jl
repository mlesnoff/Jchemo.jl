function defaults(algo::Function)

    in((detrend_pol,)).(algo) ? dump(Jchemo.ParDetrendPol()) : nothing
    in((detrend_lo,)).(algo) ? dump(Jchemo.ParDetrendLo()) : nothing
    in((detrend_asls,)).(algo) ? dump(Jchemo.ParDetrendAsls()) : nothing
    in((detrend_airpls,)).(algo) ? dump(Jchemo.ParDetrendAirpls()) : nothing
    in((detrend_arpls,)).(algo) ? dump(Jchemo.ParDetrendArpls()) : nothing
    in((fdif,)).(algo) ? dump(Jchemo.ParFdif()) : nothing
    in((interpl,)).(algo) ? dump(Jchemo.ParInterpl()) : nothing
    in((mavg,)).(algo) ? dump(Jchemo.ParMavg()) : nothing
    in((savgol,)).(algo) ? dump(Jchemo.ParSavgol()) : nothing
    in((snv,)).(algo) ? dump(Jchemo.ParSnv()) : nothing
    in((rmgap,)).(algo) ? dump(Jchemo.ParRmgap()) : nothing

    ##

    in((nipals, nipalsmiss)).(algo) ? dump(Jchemo.ParNipals()) : nothing
    in((snipals_shen, snipals_mix)).(algo) ? dump(Jchemo.ParSnipals()) : nothing

    in((pcasvd, pcaeigen, pcaeigenk, pcasph)).(algo) ? dump(Jchemo.ParPca()) : nothing
    in((pcanipals, pcanipalsmiss)).(algo) ? dump(Jchemo.ParPcanipals()) : nothing
    in((pcapp,)).(algo) ? dump(Jchemo.ParPcapp()) : nothing
    in((pcaout,)).(algo) ? dump(Jchemo.ParPcaout()) : nothing

    in((rp,)).(algo) ? dump(Jchemo.ParRp()) : nothing

    in((spca,)).(algo) ? dump(Jchemo.ParSpca()) : nothing

    in((kpca,)).(algo) ? dump(Jchemo.ParKpca()) : nothing
    in((umap,)).(algo) ? dump(Jchemo.ParUmap()) : nothing

    in((fda, fdasvd)).(algo) ? dump(Jchemo.ParFda()) : nothing

    in((blockscal,)).(algo) ? dump(Jchemo.ParBlock()) : nothing
    in((plscan, plstuck)).(algo) ? dump(Jchemo.ParPls2bl()) : nothing
    in((rasvd,)).(algo) ? dump(Jchemo.ParRasvd()) : nothing
    in((rrr,)).(algo) ? dump(Jchemo.ParRrr()) : nothing

    in((mbpca, comdim, )).(algo) ? dump(Jchemo.ParMbpca()) : nothing
    in((cca,)).(algo) ? dump(Jchemo.ParCca()) : nothing
    in((ccawold,)).(algo) ? dump(Jchemo.ParCcawold()) : nothing

    ##

    in((mlr, mlrpinv, mlrvec)).(algo) ? dump(Jchemo.ParMlr()) : nothing

    in((rr, rrchol)).(algo) ? dump(Jchemo.ParRr()) : nothing

    in((pcr,)).(algo) ? dump(Jchemo.ParPcr()) : nothing

    in((plskern, plsnipals, plswold, plsrosa, plssimp, plsravg, plsravg_unif)).(algo) ? dump(Jchemo.ParPlsr()) : nothing
    in((cglsr, aicplsr)).(algo) ? dump(Jchemo.ParCglsr()) : nothing
    in((plsrout,)).(algo) ? dump(Jchemo.ParPlsrout()) : nothing

    in((spcr,)).(algo) ? dump(Jchemo.ParSpca()) : nothing
    in((splsr,)).(algo) ? dump(Jchemo.ParSplsr()) : nothing

    in((krr,)).(algo) ? dump(Jchemo.ParKrr()) : nothing
    in((kplsr, dkplsr)).(algo) ? dump(Jchemo.ParKplsr()) : nothing

    in((svmr, svmda)).(algo) ? dump(Jchemo.ParSvm()) : nothing

    in((treer, treeda)).(algo) ? dump(Jchemo.ParTree()) : nothing
    in((rfr, rfda)).(algo) ? dump(Jchemo.ParRf()) : nothing

    in((knnr, lwmlr)).(algo) ? dump(Jchemo.ParKnn()) : nothing
    in((lwplsr, lwplsravg)).(algo) ? dump(Jchemo.ParLwplsr()) : nothing
    in((loessr,)).(algo) ? dump(Jchemo.ParLoessr()) : nothing

    in((mbplsr, mbplswest)).(algo) ? dump(Jchemo.ParMbplsr()) : nothing
    in((rosaplsr, soplsr)).(algo) ? dump(Jchemo.ParSoplsr()) : nothing

    ##

    in((dmnorm, dmnormlog)).(algo) ? dump(Jchemo.ParDmnorm()) : nothing
    in((dmkern,)).(algo) ? dump(Jchemo.ParDmkern()) : nothing

    in((lda,)).(algo) ? dump(Jchemo.ParLda()) : nothing
    in((qda,)).(algo) ? dump(Jchemo.ParQda()) : nothing
    in((rda,)).(algo) ? dump(Jchemo.ParRda()) : nothing
    in((kdeda,)).(algo) ? dump(Jchemo.ParKdeda()) : nothing
    
    in((mlrda,)).(algo) ? dump(Jchemo.ParMlrda()) : nothing
    in((rrda,)).(algo) ? dump(Jchemo.ParRrda()) : nothing
    in((plsrda, plslda)).(algo) ? dump(Jchemo.ParPlsda()) : nothing
    in((plsqda,)).(algo) ? dump(Jchemo.ParPlsqda()) : nothing
    in((plskdeda,)).(algo) ? dump(Jchemo.ParPlskdeda()) : nothing

    in((splsrda, splslda)).(algo) ? dump(Jchemo.ParSplsda()) : nothing
    in((splsqda,)).(algo) ? dump(Jchemo.ParSplsqda()) : nothing
    in((splskdeda,)).(algo) ? dump(Jchemo.ParSplskdeda()) : nothing

    in((krrda,)).(algo) ? dump(Jchemo.ParKrrda()) : nothing
    in((kplsrda, kplslda, dkplsrda, dkplslda)).(algo) ? dump(Jchemo.ParKplsda()) : nothing
    in((kplsqda, dkplsqda)).(algo) ? dump(Jchemo.ParKplsqda()) : nothing
    in((kplskdeda, dkplskdeda)).(algo) ? dump(Jchemo.ParKplskdeda()) : nothing

    in((knnda, lwmlrda)).(algo) ? dump(Jchemo.ParKnn()) : nothing
    in((lwplsrda, lwplslda)).(algo) ? dump(Jchemo.ParLwplsda()) : nothing
    in((lwplsqda,)).(algo) ? dump(Jchemo.ParLwplsqda()) : nothing

    in((mbplsrda, mbplslda)).(algo) ? dump(Jchemo.ParMbplsda()) : nothing
    in((mbplsqda,)).(algo) ? dump(Jchemo.ParMbplsqda()) : nothing
    in((mbplskdeda,)).(algo) ? dump(Jchemo.ParMbplskdeda()) : nothing

    ##

    in((occstah,)).(algo) ? dump(Jchemo.ParOccstah()) : nothing
    in((occsd, occod, occsdod)).(algo) ? dump(Jchemo.ParOcc()) : nothing
    in((occknn, occlknn)).(algo) ? dump(Jchemo.ParOccknn()) : nothing

end
