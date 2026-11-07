function defaults(algo::Function)

    defs = [

        detrend_pol Jchemo.ParDetrendpol ;
        detrend_lo Jchemo.ParDetrendlo ;
        detrend_asls Jchemo.ParDetrendasls{Float64} ;
        detrend_airpls Jchemo.ParDetrendairpls{Float64} ;
        detrend_arpls Jchemo.ParDetrendarpls{Float64} ;
        fdif Jchemo.ParFdif ;
        interpl Jchemo.ParInterpl ;
        mavg Jchemo.ParMavg ;
        emsc Jchemo.ParEmsc ;
        savgol Jchemo.ParSavgol ;
        snv Jchemo.ParSnv ;
        rmgap Jchemo.ParRmgap ;
        scale Jchemo.ParScale ;
        cscale Jchemo.ParScale ;

        nipals Jchemo.ParNipals ;
        nipalsmiss Jchemo.ParNipals ;
        snipals_shen Jchemo.ParSnipals ;

        pcasvd Jchemo.ParPca ; 
        pcaeigen Jchemo.ParPca ; 
        pcaeigenk Jchemo.ParPca ; 
        pcasph Jchemo.ParPca ;
        pcanipals Jchemo.ParPcanipals ; 
        pcanipalsmiss Jchemo.ParPcanipals ;
        pcapp Jchemo.ParPcapp ;
        pcaout Jchemo.ParPcaout ;

        covsel Jchemo.ParCovsel ;
        
        rp Jchemo.ParRp ;

        spca Jchemo.ParSpca ;

        kpca Jchemo.ParKpca ;
        umap Jchemo.ParUmap ;

        fda Jchemo.ParFda ; 
        fdasvd Jchemo.ParFda ;

        blockscal Jchemo.ParBlock ;
        plscan Jchemo.ParPls2bl ; 
        plstuck Jchemo.ParPls2bl ;
        rasvd Jchemo.ParRasvd ;
        rrr Jchemo.ParRrr ;

        cpca Jchemo.ParCpca ; 
        comdim Jchemo.ParCpca ;
        cca Jchemo.ParCca ;
        ccawold Jchemo.ParCcawold ;

        ##

        mlr Jchemo.ParMlr ; 

        rr Jchemo.ParRr ; 
        rrchol Jchemo.ParRr ;

        pcr Jchemo.ParPcr ;

        plskern Jchemo.ParPlsr ; 
        plsnipals Jchemo.ParPlsr ; 
        plswold Jchemo.ParPlswold ; 
        plsrosa Jchemo.ParPlsr ; 
        plssimp Jchemo.ParPlsr ; 
        plsravg Jchemo.ParPlsravg ; 
        plsravg_unif Jchemo.ParPlsravgunif ;
        cglsr Jchemo.ParCglsr ; 
        aicplsr Jchemo.ParCglsr ;
        plsrout Jchemo.ParPlsrout ;

        spcr Jchemo.ParSpca ;
        splsr Jchemo.ParSplsr ;

        mwplsr Jchemo.ParMwplsr ;

        krr Jchemo.ParKrr ;
        kplsr Jchemo.ParKplsr ; 
        dkplsr Jchemo.ParKplsr ;

        svmr Jchemo.ParSvm ; 
        svmda Jchemo.ParSvm ;

        treer Jchemo.ParTree ; 
        treeda Jchemo.ParTree ;
        rfr Jchemo.ParRf ; 
        rfda Jchemo.ParRf ;

        knnr Jchemo.ParKnn ;
        lwmlr Jchemo.ParLwmlr ;
        lwplsr Jchemo.ParLwplsr ; 
        lwplsravg Jchemo.ParLwplsravg ;
        loessr Jchemo.ParLoessr ;

        protoplsr Jchemo.ParProtoplsr{Float64} ;
        protoclustplsr Jchemo.ParProtoclustplsr ;

        mbplsr Jchemo.ParMbplsr ; 
        mbplswest Jchemo.ParMbplsr ;
        rosaplsr Jchemo.ParSoplsr ; 
        soplsr Jchemo.ParSoplsr ;

        ##

        dmnorm Jchemo.ParDmnorm ; 
        dmnormlog Jchemo.ParDmnorm ;
        dmkern Jchemo.ParDmkern ;

        lda Jchemo.ParLda{Float64} ;
        qda Jchemo.ParQda ;
        rda Jchemo.ParRda ;
        kdeda Jchemo.ParKdeda ;
        
        mlrda Jchemo.ParMlrda{Float64} ;
        rrda Jchemo.ParRrda ;
        plsrda Jchemo.ParPlsda{Float64} ;
        plslda Jchemo.ParPlsda{Float64} ;
        plsqda Jchemo.ParPlsqda ;
        plskdeda Jchemo.ParPlskdeda ;

        splsrda Jchemo.ParSplsda ; 
        splslda Jchemo.ParSplsda ;
        splsqda Jchemo.ParSplsqda ;
        splskdeda Jchemo.ParSplskdeda ;

        mwplsqda Jchemo.ParMwplsqda ;

        krrda Jchemo.ParKrrda ;
        kplsrda Jchemo.ParKplsda ; 
        kplslda Jchemo.ParKplsda ; 
        dkplsrda Jchemo.ParKplsda ; 
        dkplslda Jchemo.ParKplsda ;
        kplsqda Jchemo.ParKplsqda ;
        dkplsqda Jchemo.ParKplsqda ;
        kplskdeda Jchemo.ParKplskdeda ; 
        dkplskdeda Jchemo.ParKplskdeda ;

        knnda Jchemo.ParKnn ;
        lwmlrda Jchemo.ParLwmlr ;
        lwplsrda Jchemo.ParLwplsda ; 
        lwplslda Jchemo.ParLwplsda ;
        lwplsqda Jchemo.ParLwplsqda ;

        mbplsrda Jchemo.ParMbplsda{Float64} ;
        mbplslda Jchemo.ParMbplsda{Float64} ;
        mbplsqda Jchemo.ParMbplsqda ;
        mbplskdeda Jchemo.ParMbplskdeda ;

        soplsrda Jchemo.ParSoplsda{Float64} ;
        soplslda Jchemo.ParSoplsda{Float64} ;
        soplsqda Jchemo.ParSoplsqda ;
        soplskdeda Jchemo.ParSoplskdeda ;

        ##

        occsd Jchemo.ParOcc ; 
        occod Jchemo.ParOcc ;
        occsdod Jchemo.ParOccsdod ;
        occdds Jchemo.ParOccdds ;
        occstah Jchemo.ParOccstah ;
        occknn Jchemo.ParOccknn ; 
        occlknn Jchemo.ParOccknn ;
        occmwpca Jchemo.ParOccmwpca ;
        occmwpls Jchemo.ParOccmwpls 

        ]

    s = in((algo,)).(defs[:, 1]) 
    first(defs[s, 2])

end
