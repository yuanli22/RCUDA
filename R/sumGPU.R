#' sumGPU
#'
#' Compute the sum of given vector  
#' 
#' This function computes sum of given vector by using CUDA vector reduction
#' @param x list consisting of R external GPU pointer and dimension 
#' @return vector sum
#' @author Yuan Li        
#' @keywords GPU 
#' @seealso \code{\link{gatherGPU}} \code{\link{createGPU}} 
#' @export
#' @examples
#' a <- createGPU(1:4)
#' sumGPU(a)

sumGPU<-function(x)
{
 checkGPU(x) 
 ext<-.Call(
                "vector_reduction",
                x$ptr,      
                as.integer(x[2])*as.integer(x[3]),
                PACKAGE= "supplement" 
              )
 
    return(ext)

}