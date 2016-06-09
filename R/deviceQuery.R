#' gpuQuery
#'
#' This function returns the information of available GPU device in system
#' @seealso \code{\link{creategpu}} 
#' @export
#' @examples
#' gpuQuery()


gpuQuery<-function()
{
    ext<-.Call(
                "devicequery",
		  PACKAGE= "supplement"
              )
}
