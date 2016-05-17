#' maxGPU
#'
#' finds the (smallest) index of the element with the maximum magnitude of given vector
#' This function finds the (smallest) index of the element with the maximum magnitude of given vector
#' by using CUDA cublas function cublasIdamin
#' @param input list consisting of R external GPU pointer and dimension 
#' @return the resulting index 
#' @seealso \code{\link{minGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' maxGPU(a_gpu)

maxGPU<-function(input)
{
    checkGPU(input)
    ext <- .Call(
                "maxGPU",
                input$ptr,              
                as.integer(input[2])*as.integer(input[3])
              )
    return(ext)
}