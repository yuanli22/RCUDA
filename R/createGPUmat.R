#' createGPUmat
#'
#' Create a GPU matrix by copying from the input R vector 
#' 
#' This function creates a matrix in GPU by calling the CUDA cudamalloc function, and then copys
#' the values of matrix from input R vector (matrix stored in column-major format). The output is a list consisting of the GPU pointer and matrix dimension 
#' @param input R vector (matrix stored in column-major format) to be copied (numerical)
#' @param nrow number of rows
#' @param ncol number of columns
#' @return a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @note output is a R external GPU pointer and can only be used in Rcublas functions
#' @author Yuan Li        
#' @keywords GPU matrix create
#' @seealso \code{\link{gatherGPU}} \code{\link{createGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' am_gpu <- createGPUmat(a, 2, 2)
#' gatherGPU(am_gpu) 

createGPUmat<-function(input, nrow, ncol)
{
    n <- length(input)
    if (n!=nrow*ncol)
    stop("dimension wrong")
    ext<-.Call(
                "createGPU",
                as.numeric(input),             
                as.integer(n)
)

    ext <- GPUobject(ext, nrow, ncol)
    gc()
    return(ext)
}
