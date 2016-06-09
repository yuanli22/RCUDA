#' GPUobject
#'
#' classify the input as GPU vector and assign its dimension
#'
#' This function classifies the input object as GPU vector and assign its dimension.
#' The output of this function is a list consisting of the GPU pointer and its dimension. 
#' @param input R external pointer
#' @param length1 vector length
#' @param length2 always 1 as vector
#' @return a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{vector length}
#' \item{n: }{always 1 as vector}
#' }
#' @note output is a R external GPU pointer and can only be used in Rcublas functions
#' @author Yuan Li        
#' @keywords GPU vector create
#' @seealso \code{\link{gatherGPU}} 


GPUobject <- function(input, length1, length2)
{
    obj <- list( ptr = input,
                 m = length1,
                 n = length2)
    class(obj) <- append(class(obj), "GPUvector")
    return (obj)
               
}

checkGPU <- function(input)
{
    if (inherits(input, "GPUvector") == "FALSE")
    stop("input is not a GPUvector")
}
