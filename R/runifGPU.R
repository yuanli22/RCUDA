
#' standard uniform random number generator
#'
#' This function generates uniformly distributed numbers between 0 and 1 
#' by using CUDA curand function CURAND_RNG_PSEUDO_DEFAULT and curandGenerateUniformDouble
#' @param n number of random numbers 
#' @param seed random number generator seed
#' @return random numbers vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix X's number of rows}
#' \item{n: }{matrix X's number of columns}
#' }
#' @seealso \code{\link{createGPU}} \code{\link{createGPUmat}} 
#' @export
#' @examples
#' a_gpu <- uniformRNGGPU(100, 15)
#' gatherGPU(a_gpu)



uniformRNGGPU<-function(n, seed=1)
{

    ext<-.Call(
                "uniformRNGGPU",                        
                as.integer(n),
                as.numeric(seed)
              )
    ext<-GPUobject(ext, as.integer(n),as.integer(1))
    return(ext)

}

