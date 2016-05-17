#' norm2GPU
#'
#' Compute the Euclidean norm of given vector 
#' 
#' This function computes Euclidean norm of given vector by using CUDA cublas function cublasDnrm2
#' @param input list consisting of R external GPU pointer and dimension 
#' @return vector Euclidean norm, a non-negative number.
#' @author Yuan Li        
#' @keywords GPU 
#' @seealso \code{\link{gatherGPU}} \code{\link{createGPU}} 
#' @export
#' @examples
#' a <- 1:4
#' a_gpu <- createGPU(a)
#' norm2GPU(a_gpu) 


norm2GPU<-function(input)
{
    checkGPU(input)
    ext<-.Call("norm2GPU",
                input$ptr,              
                as.integer(input[2])*as.integer(input[3])
               )
    return(ext)

}
