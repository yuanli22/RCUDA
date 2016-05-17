
#' minGPU
#'
#' finds the (smallest) index of the element with the minimum magnitude of given vector
#' This function finds the (smallest) index of the element with the minimum magnitude of given vector
#' by using CUDA cublas function cublasIdamin
#' @param input list consisting of R external GPU pointer and dimension 
#' @return the resulting index 
#' @seealso \code{\link{maxGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' minGPU(a_gpu) 


minGPU<-function(input)
{
    checkGPU(input)
    ext<-.Call(
                "minGPU",
                input$ptr,              
                as.integer(input[2])*as.integer(input[3])
              )
    return(ext)

}
