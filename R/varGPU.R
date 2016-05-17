#' varGPU
#'
#' Compute the variance of given sample  
#' 
#' This function computes variance of given sample by using CUDA vector reduction
#' @param x list consisting of R external GPU pointer and dimension 
#' @return sample variance
#' @author Yuan Li        
#' @keywords GPU 
#' @seealso \code{\link{sumGPU}}
#' @export
#' @examples
#' a <- createGPU(1:4)
#' varGPU(a)

varGPU<-function(x)
{
 checkGPU(x) 
 n <- as.integer(x[2])*as.integer(x[3])
 c <- createGPU(1:n)
 mean <- sumGPU(x)/n
 ext <- .Call(
                "cudavarGPU",
                x$ptr, 
                c$ptr,
                n,
                as.numeric(mean),
                PACKAGE= "supplement" 
              )
 ext<-GPUobject(ext, as.integer(x[2]),as.integer(x[3]))
 result <- sumGPU(ext)/(n-1) 
 return(result)

}