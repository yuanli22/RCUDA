#' GPUobject
#'
#' classify the input as GPU vector/matrix and assign its dimension
#'
#' This function classifies the input object as GPU vector/matrix
#' and assign its dimension
#' The output of this function is a list consisting of the GPU pointer
#' and its dimension
#' @param input R external pointer
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
#' @keywords GPU vector create
#' @seealso \code{\link{gathergpu}} 


GPUobject <- function(input, nrow, ncol)
{
    obj <- list(ptr = input,
                m   = nrow,
                n   = ncol)
    class(obj) <- append(class(obj), "GPUvector")
    return (obj)              
}

checkGPU <- function(input)
{
    if (inherits(input, "GPUvector") == "FALSE")
    	stop("input is not a GPUvector")
}
