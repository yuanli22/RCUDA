#' creategpu
#'
#' Create a GPU vector/matrix by copying from the input R vector 
#' 
#' This function creates a vector/matrix in GPU by calling the CUDA 
#' cudamalloc function, and then copys from input R vector.
#' The output of this function is a list consisting of the GPU pointer and 
#' its dimension. 
#' @param input R vector to be copied  
#' @param nrow the desired number of rows
#' @param ncol the desired number of columns  
#' @details If either one of nrow or ncol is not given, an one column matrix/vector is 
#' returned. This function returns row-major matrix.
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
#' @export
#' @examples
#' a <- rnorm(6)
#' a_gpu <- creategpu(a, 2, 3)
#' gathergpu(a_gpu) 

creategpu <- function(input, nrow = NULL, ncol = NULL)
{
    n <- length(input)
    if (!is.null(nrow) && !is.null(ncol)) { 
    	if (n != (as.integer(nrow) * as.integer(ncol)))
           stop ("dimension does not match")
    }
    ext <- .Call(
                "createGPU",
                as.numeric(input),             
                as.integer(n)
                )
    if (is.null(nrow) || is.null(ncol)) {
    	ext <- GPUobject(ext, n ,1)
    } else {  
      if ((length(nrow) != 1) || length(ncol) != 1)
         stop ("dimension need to be scalar")
    ext <- GPUobject(ext, as.integer(nrow), as.integer(ncol))
    }
    return(ext)
}

#' gathergpu
#'
#' Copy GPU matrix/vector to R vector 
#' 
#' This function copys GPU vector/matrix to R vector
#' @param input list consisting of R external GPU pointer and its dimension 
#' @details The output is always R vector, 
#' and GPU matrix will be copied by row-major. For example, an m by n
#' GPU matrix will be converted to a m*n R vector.
#' @return R vector
#' @note output is R vector and can be used by any R functions
#' @author Yuan Li        
#' @keywords GPU 
#' @seealso \code{\link{gathergpu}} \code{\link{creategpu}} 
#' @export
#' @examples
#' a <- 1:6
#' am_gpu <- creategpu(a, 3, 2)
#' gathergpu(am_gpu) 

gathergpu <- function(input)
{
    checkGPU(input)
    ext<-.Call(
                "gatherGPU",
                input$ptr,             
                as.integer(input[2]) * as.integer(input[3])
              )
    return(ext)
}



