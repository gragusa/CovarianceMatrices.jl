using PkgBenchmark
using CovarianceMatrices


@benchgroup "HAC - Fixed Bandwidth" begin
    @benchgroup "No prewithen" begin
        nf = string(Pkg.dir("CovarianceMatrices"), "/test/ols_hac.csv")
        df = readtable(nf)
        lm1 = glm(y~x+w, df, Normal(), IdentityLink())
        @bench "Truncated Kernel" vcov(lm1, TruncatedKernel(1.0), prewhite = false)
        @bench "Quadratic Spectral Kernel" vcov(lm1, QuadraticSpectralKernel(1.0), prewhite = false)
        @bench "Parzen Kernel" vcov(lm1, ParzenKernel(1.0), prewhite = false)
        @bench "Tukey Hanning Kernel" vcov(lm1, TukeyHanningKernel(1.0), prewhite = false)
    end

    @benchgroup "Prewithen" begin
        nf = string(Pkg.dir("CovarianceMatrices"), "/test/ols_hac.csv")
        df = readtable(nf)
        lm1 = glm(y~x+w, df, Normal(), IdentityLink())
        @bench "Truncated Kernel" vcov(lm1, TruncatedKernel(1.0), prewhite = true)
        @bench "Quadratic Spectral Kernel" vcov(lm1, QuadraticSpectralKernel(1.0), prewhite = true)
        @bench "Parzen Kernel" vcov(lm1, ParzenKernel(1.0), prewhite = true)
        @bench "Tukey Hanning Kernel" vcov(lm1, TukeyHanningKernel(1.0), prewhite = true)
    end

end


@benchgroup "HAC - Optimal Bandwidth" begin
    @benchgroup "No prewithen" begin
        nf = string(Pkg.dir("CovarianceMatrices"), "/test/ols_hac.csv")
        df = readtable(nf)
        lm1 = glm(y~x+w, df, Normal(), IdentityLink())
        @bench "Truncated Kernel" vcov(lm1, TruncatedKernel(), prewhite = false)
        @bench "Quadratic Spectral Kernel" vcov(lm1, QuadraticSpectralKernel(), prewhite = false)
        @bench "Parzen Kernel" vcov(lm1, ParzenKernel(), prewhite = false)
        #@bench "Tukey Hanning Kernel" vcov(lm1, TukeyHanningKernel(), prewhite = false)
    end

    @benchgroup "Prewithen" begin
        nf = string(Pkg.dir("CovarianceMatrices"), "/test/ols_hac.csv")
        df = readtable(nf)
        lm1 = glm(y~x+w, df, Normal(), IdentityLink()) 
        @bench "Truncated Kernel" vcov(lm1, TruncatedKernel(), prewhite = true)
        @bench "Quadratic Spectral Kernel" vcov(lm1, QuadraticSpectralKernel(), prewhite = true)
        @bench "Parzen Kernel" vcov(lm1, ParzenKernel(), prewhite = true)
        #@bench "Tukey Hanning Kernel" vcov(lm1, TukeyHanningKernel(), prewhite = true)
    end
end
