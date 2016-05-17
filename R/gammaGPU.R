#' gammaGPU
#'
#' This function apply the gammma function to a given vector or matrix
#' by using CUDA function
#' @param input (non-negative vector) list consisting of R external GPU pointer and dimension 
#' @return gamma vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{expGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' gammaGPU(a_gpu)->b_gpu
#' gatherGPU(b_gpu)


gammaGPU <- function (input) 
{ 

    checkGPU(input)
    ext<-.Call("vector_gamma", 
                input$ptr,
                as.integer(input[2])*as.integer(input[3]),
                PACKAGE= "supplement"

               )
    ext<-GPUobject(ext, as.integer(input[2]),as.integer(input[3]))
    return(ext)

  }
