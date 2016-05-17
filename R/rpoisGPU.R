
#' Poisson random number generator
#'
#' This function generates Poisson distributed numbers 
#' by using CUDA curand function CURAND_RNG_PSEUDO_DEFAULT and curandGeneratePoisson
#' @param n number of random numbers 
#' @param lambda mean of Poisson distribution; default value 1 
#' @param seed random number generator seed
#' @return random numbers vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix X's number of rows}
#' \item{n: }{matrix X's number of columns}
#' }
#' @seealso \code{\link{normRNGGPU}} 
#' @export
#' @examples
#' a_gpu <- poissonRNGGPU(100, 1)
#' gatherGPU(a_gpu)


##need test
poissonRNGGPU<-function(n, lambda=1, seed=1)
{

    ext<-.Call(
                "poissonRNGGPU",                        
                as.integer(n),
	         as.numeric(lambda),
	         as.numeric(seed)
              )
    ext<-GPUobject(ext, as.integer(n),as.integer(1))
    return(ext)

}

