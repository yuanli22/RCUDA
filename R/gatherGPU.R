
#' gatherGPU
#'
#' Copy GPU vector to R vector 
#' 
#' This function copys GPU vector/matrix to R vector
#' @param input list consisting of R external GPU pointer and dimension 
#' @return R vector
#' @note output is R vector and can be used by any R functions
#' @author Yuan Li        
#' @keywords GPU 
#' @seealso \code{\link{gatherGPU}} \code{\link{createGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' am_gpu <- createGPUmat(a, 2, 2)
#' gatherGPU(am_gpu) 

gatherGPU<-function(input)
{
    checkGPU(input)
    ext<-.Call(
                "gatherGPU",
                input$ptr,             
                as.integer(input[2])*as.integer(input[3])
              )
    return(ext)

}

