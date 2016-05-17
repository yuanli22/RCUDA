
#' powerGPU
#'
#' This function computes the power of given vector or matrix 
#' by using CUDA function 
#' @param input list consisting of R external GPU pointer and dimension 
#' @param alpha power factor
#' @return powered vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{sqrtGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2
#' a_gpu <- createGPU(a)
#' powerGPU(a_gpu, b)->b_gpu
#' gatherGPU(b_gpu)


powerGPU <- function (input, alpha=1) 
{
    checkGPU(input)
    ext<-.Call("vector_power", 
                input$ptr,
                as.integer(input[2])*as.integer(input[3]),
                as.numeric(alpha),
                PACKAGE= "supplement"

               )
    ext<-GPUobject(ext, as.integer(input[2]),as.integer(input[3]))
    return(ext)

  }
 
