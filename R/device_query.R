#' gpuquery
#'  
#' This function returns the information of available GPU device in system
#' @seealso \code{\link{creategpu}} 
#' @export
#' @examples
#' gpuquery()

gpuquery <- function()
{
    ext <- .Call(
                "devicequery",
		  PACKAGE = "supplement"
              )
}
