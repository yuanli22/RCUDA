#' betaGPU
#'
#' This function apply the beta function to a given vector or matrix
#' by using CUDA function
#' @param input (non-negative vector) list consisting of R external GPU pointer and dimension 
#' @return beta result, a list consisting of
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
#' betaGPU(a_gpu)->b_gpu
#' gatherGPU(b_gpu)


betaGPU <- function (input1, input2) 
{ 
    checkGPU(input1)
    checkGPU(input2)
    if (as.integer(input1[2])*as.integer(input1[3])!=as.integer(input2[2])*as.integer(input2[3]))
    stop ("vectors dimension don't match")
    ext<-.Call("vector_beta", 
                input1$ptr,
                input2$ptr,
                as.integer(input1[2])*as.integer(input1[3]),
                PACKAGE= "supplement"

               )
    ext<-GPUobject(ext, as.integer(input1[2]),as.integer(input1[3]))
    return(ext)

  }
