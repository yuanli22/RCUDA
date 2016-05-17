
#' expGPU
#'
#' This function computes the exponential of given vector or matrix
#' by using CUDA function
#' @param input list consisting of R external GPU pointer and dimension 
#' @return exponential vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{scaleGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' expGPU(a_gpu)->b_gpu
#' gatherGPU(b_gpu)



expGPU <- function (input) 
{ 
    checkGPU(input)   
    ext<-.Call("vector_exp", 
                input$ptr,
                as.integer(input[2])*as.integer(input[3]),
                PACKAGE= "supplement"

               )
    ext<-GPUobject(ext, as.integer(input[2]),as.integer(input[3]))
    return(ext)
  }
