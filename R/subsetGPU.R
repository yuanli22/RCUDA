
#' subsetGPU
#'
#' This function select and copy subset of a given GPU vector
#' by using CUDA function
#' @param input list consisting of R external GPU pointer and dimension
#' @param index index of the vector subset 
#' @return subset of vector, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{createGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' subsetGPU(a_gpu,c(1,2))->b_gpu
#' gatherGPU(b_gpu)



subsetGPU <- function (input, index) 
{ 
    checkGPU(input)
    n <- length(index)   
    ext <- .Call("subset_GPU", 
                input$ptr,
                as.integer(n),
                index,
                PACKAGE= "supplement"

               )
    ext<-GPUobject(ext, as.integer(n),1)
    return(ext)
  }
