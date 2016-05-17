#' createGPU
#'
#' Create a GPU vector by copying from the input R vector 
#' 
#' This function creates a vector in GPU by calling the CUDA cudamalloc function, and then copys
#' the values of vector from input R vector. The output of this 
#' function is a list consisting of the GPU pointer and vector length . 
#' @param input R vector to be copied (numerical)
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
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' gatherGPU(a_gpu) 

createGPU<-function(input)
{

    n<-length(input)
    ext<-.Call(
                "createGPU",
                as.numeric(input),             
                as.integer(n)
)

    ext <- GPUobject(ext, n ,1)
    return(ext)
}


