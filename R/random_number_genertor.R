#' normal random number generator
#'
#' This function generates normally distributed numbers 
#' by using CUDA curand function CURAND_RNG_PSEUDO_DEFAULT and curandGenerateNormalDouble
#' @param n number of random numbers 
#' @param mean mean of normal distribution; default value 0 
#' @param sd standard deviation of normal distribution; default value 1 
#' @param seed random number generator seed
#' @return random numbers vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix X's number of rows}
#' \item{n: }{matrix X's number of columns}
#' }
#' @seealso \code{\link{lognormRNGGPU}}  
#' @export
#' @examples
#' a_gpu <- normRNGGPU(100, 0, 1, 15)
#' gatherGPU(a_gpu)


normRNGGPU<-function(n, mean=0, sd=1, seed=1)
{

    ext<-.Call(
                "normRNGGPU",                        
                as.integer(n),
	         as.numeric(mean),
	         as.numeric(sd),
	         as.numeric(seed)
              )
    ext<-GPUobject(ext, as.integer(n),as.integer(1))
    return(ext)

}



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




