#' scaleGPU
#'
#' This function scales the given vector by a scalar
#' by using CUDA cublas function cublasDcopy
#' @param input list consisting of R external GPU pointer and dimension 
#' @param alpha scale factor
#' @return scaled vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{expGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2
#' a_gpu <- createGPU(a)
#' scaleGPU(a_gpu, b)->b_gpu
#' gatherGPU(b_gpu)


scaleGPU<-function(input, alpha)
{
    checkGPU(input)
    ext<-.Call(
                "scaleGPU",
                input$ptr,                        
                as.integer(input[2])*as.integer(input[3]),
                as.numeric(alpha)
              )
     ext<-GPUobject(ext, as.integer(input[2]),as.integer(input[3]))
    return(ext)
}