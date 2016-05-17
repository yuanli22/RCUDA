
#' log normal random number generator
#'
#' This function generates log-normally distributed numbers 
#' by using CUDA curand function CURAND_RNG_PSEUDO_DEFAULT and curandGenerateLogNormalDouble
#' @param n number of random numbers 
#' @param mean mean of log-normal distribution; default value 0 
#' @param sd standard deviation of log-normal distribution; default value 1 
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
#' a_gpu <- lognormRNGGPU(100, 0, 1, 15)
#' gatherGPU(a_gpu)

lognormRNGGPU<-function(n, mean=0, sd=1, seed=1)
{

    ext<-.Call(
                "lognormRNGGPU",                        
                as.integer(n),
	         as.numeric(mean),
	         as.numeric(sd),
	         as.numeric(seed)
              )
    ext<-GPUobject(ext, as.integer(n),as.integer(1))
    return(ext)

}
