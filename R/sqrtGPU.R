
#' sqrtGPU
#'
#' This function computes the square root of given vector or matrix
#' by using CUDA function
#' @param input (non-negative vector) list consisting of R external GPU pointer and dimension 
#' @return square root vector, a list consisting of
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
#' sqrtGPU(a_gpu)->b_gpu
#' gatherGPU(b_gpu)


sqrtGPU <- function (input) 
{ 

    checkGPU(input)
    createGPU(1:as.integer(input[2])*as.integer(input[3]))->c
    ext<-.Call("vector_sqrt", 
                input$ptr,
                c$ptr,
                as.integer(input[2])*as.integer(input[3]),
                PACKAGE= "supplement"

               )
    ext<-GPUobject(ext, as.integer(input[2]),as.integer(input[3]))
    return(ext)

  }
