#' gemvgpu
#'
#' This function computes matrix-vector multipication y = a A x + b y
#' by using CUDA cublas function cublasDgemv
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param x input vector; list of R external GPU pointer and dimension
#' @param y input/output vector; list of R external GPU pointer and dimension
#' @param alpha scale factor a of matrix A; default 1
#' @param beta scale factor b of vector y; default 0
#' @param trans matrix A transpose operator, 1 (non-transpose), 2 (transpose),
#' 3 (conjugate transpose); default at 1 (non-transpose)
#' @return vector y, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{length of vector y}
#' \item{n: }{1}
#' }
#' @seealso \code{\link{gergpu}} 
#' @export
#' @examples
#' A <- 1:12
#' x <- 1:3
#' y <- 1:4
#' A_gpu <- creategpu(A, 4, 3)
#' x_gpu <- creategpu(x)
#' y_gpu <- creategpu(y)
#' gemvgpu(trans = 1, alpha = 1, A_gpu, x_gpu, beta = 1, y_gpu)
#' gathergpu(y_gpu)

gemvgpu <- function(trans = 1, alpha = 1, A, x, beta = 0, y)
{
  checkGPU(A)
  checkGPU(x)
  checkGPU(y)
  if ((trans != 1) && (trans != 2) && (trans != 3))
    stop ("transpose operation input error")
  if (!is.numeric(beta) || !is.numeric(alpha))
    stop ("scale factor should be numerical")
  if (trans == 1) {
    if (as.integer(A[3]) != as.integer(x[2]))
      stop ("A x dimension doesn't match")
    if (as.integer(A[2]) != as.integer(y[2]))
      stop ("A y dimension doesn't match")
    } 
  if ((trans == 2) || (trans == 3)){
    if (as.integer(A[2]) != as.integer(x[2]))
      stop ("A x dimension doesn't match")
    if (as.integer(A[3]) != as.integer(y[2]))
      stop ("A y dimension doesn't match")
    }
  ext <- .Call(
                "gemvGPU",
                 A$ptr,
                 x$ptr,
                 y$ptr,
		   as.numeric(alpha),
                 as.numeric(beta),
                 as.integer(A[2]),
                 as.integer(A[3]),
		   as.numeric(trans)            
               )
   ext <- GPUobject(ext, as.integer(y[2]), 1)
   return(ext)
}


#' gbmvgpu
#'
#' This function computes banded matrix-vector multipication y = a A x + b y
#' by using CUDA cublas function cublasDgbmv
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param x input vector; list of R external GPU pointer and dimension
#' @param y input/output vector; list of R external GPU pointer and dimension
#' @param kl number of subdiagonals
#' @param ku number of superdiagonals
#' @param alpha scale factor a of banded matrix A; default 1
#' @param beta scale factor b of vector y; default 0
#' @param trans matrix A transpose operator, 1 (non-transpose), 2 (transpose),
#' 3 (conjugate transpose); default at 1 (non-transpose)
#' @return vector y, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{length of vector y}
#' \item{n: }{1}
#' }
#' @seealso \code{\link{gergpu}} 
#' @export
#' @examples
#' A <- 1:12
#' x <- 1:3
#' y <- 1:4
#' A_gpu <- creategpu(A, 4, 3)
#' x_gpu <- creategpu(x)
#' y_gpu <- creategpu(y)
#' gemvgpu(A_gpu, x_gpu, y_gpu, 1, 1, 1)
#' gathergpu(y_gpu)

gbmvgpu <- function(trans = 1, kl, ku, alpha = 1, A, x, beta = 0, y)
{
  checkGPU(A)
  checkGPU(x)
  checkGPU(y)
  if ((trans != 1) && (trans != 2) && (trans != 3))
    stop ("transpose operation input error")
  if (!is.numeric(beta) || !is.numeric(alpha))
    stop ("scale factor should be numerical")
  if (trans == 1) {
    if (as.integer(A[3]) != as.integer(x[2]))
      stop ("A x dimension doesn't match")
    if (as.integer(A[2]) != as.integer(y[2]))
      stop ("A y dimension doesn't match")
    } 
  if ((trans == 2) || (trans == 3)){
    if (as.integer(A[2]) != as.integer(x[2]))
      stop ("A x dimension doesn't match")
    if (as.integer(A[3]) != as.integer(y[2]))
      stop ("A y dimension doesn't match")
    }
  ext <- .Call(
                "gbmvGPU",
                 A$ptr,
                 x$ptr,
                 y$ptr,
		   as.numeric(alpha),
                 as.numeric(beta),
                 as.integer(A[2]),
                 as.integer(A[3]),
		   as.integer(kl),
	          as.integer(ku),
		   as.numeric(trans)            
               )
   ext <- GPUobject(ext, as.integer(y[2]), 1)
   return(ext)
}


#' gergpu
#'
#' This function perform the the rank-1 update A = a x y T + A,
#' by using CUDA cublas function cublasDger
#' @param A input/output matrix; list of R external GPU pointer and dimension 
#' @param x input vector; list of R external GPU pointer and dimension
#' @param y input vector; list of R external GPU pointer and dimension
#' @param alpha scale factor a of matrix A; default 1
#' @return updated matrix A, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix A's number of rows}
#' \item{n: }{matrix A's number of columns}
#' }
#' @seealso \code{\link{gemvgpu}} 
#' @export
#' @examples
#' A <- 1:12
#' x <- 1:3
#' y <- 1:4
#' A_gpu <- creategpu(A, 3, 4)
#' x_gpu <- creategpu(x)
#' y_gpu <- creategpu(y)
#' gergpu(1,x_gpu, y_gpu, A_gpu)
#' gathergpu(A_gpu)

gergpu <- function(alpha = 1, x, y, A)
{
  checkGPU(A)
  checkGPU(x)
  checkGPU(y)
  if (as.integer(A[2]) != as.integer(x[2]))
     stop ("A x dimension doesn't match")
  if (as.integer(A[3]) != as.integer(y[2]))
     stop ("A y dimension doesn't match")
  ext <- .Call(
                "gerGPU",
                 A$ptr,
                 x$ptr,
                 y$ptr,
		   as.numeric(alpha),
                 as.integer(A[2]),
                 as.integer(A[3])         
               )
   ext <- GPUobject(ext, as.integer(A[2]), as.integer(A[3]))
   return(ext)
}


#' sbmvgpu
#'
#' This function computes symmetric banded matrix-vector multipication y = a A x + b y
#' by using CUDA cublas function cublasDsbmv
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param x input vector; list of R external GPU pointer and dimension
#' @param y input/output vector; list of R external GPU pointer and dimension
#' @param k number of subdiagonals
#' @param alpha scale factor a of symmetric banded matrix A; default 1
#' @param beta scale factor b of vector y; default 0
#' @param fillmode indicates if matrix A lower or upper part is stored, 
#' the other symmetric part is not referenced and is inferred from the 
#' stored elements. if fillmode == 1 then the symmetric 
#' banded matrix A is stored column by column, with the main diagonal of 
#' the matrix stored in row 1, the first subdiagonal in row 2 
#' (starting at first position), 
#' the second subdiagonal in row 3 (starting at first position), etc. 
#' if fillmode == 2 then the symmetric banded matrix A is 
#' stored column by column, with the main diagonal of the matrix stored 
#' in row k+1, the first superdiagonal in row k (starting at second position),
#' the second superdiagonal in row k-1 (starting at third position), etc.
#' @return vector y, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{length of vector y}
#' \item{n: }{1}
#' }
#' @seealso \code{\link{gemvgpu}} 
#' @export
 

sbmvgpu <- function(fillmode = 1, k, alpha = 1, A, x, beta = 0, y)
{
  checkGPU(A)
  checkGPU(x)
  checkGPU(y)
  if (as.integer(A[2]) != as.integer(A[3]))
     stop ("A needs to be square matrix")
  if (as.integer(A[3]) != as.integer(x[2]))
     stop ("A x dimension doesn't match")
  if (as.integer(A[2]) != as.integer(y[2]))
     stop ("A y dimension doesn't match") 
  ext <- .Call(
                "sbmvGPU",
                 A$ptr,
                 x$ptr,
                 y$ptr,
		   as.numeric(alpha),
                 as.numeric(beta),
                 as.integer(A[2]),
		   as.integer(k),
		   as.numeric(fillmode)            
               )
   ext <- GPUobject(ext, as.integer(y[2]), 1)
   return(ext)
}

#' symvgpu
#'
#' This function computes symmetric matrix-vector multipication y = a A x + b y
#' by using CUDA cublas function cublasDsymv
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param x input vector; list of R external GPU pointer and dimension
#' @param y input/output vector; list of R external GPU pointer and dimension
#' @param alpha scale factor a of symmetric banded matrix A; default 1
#' @param beta scale factor b of vector y; default 0
#' @param fillmode indicates if matrix A lower or upper part is stored, 
#' the other symmetric part is not referenced and is inferred from the 
#' stored elements. if fillmode == 1 then the symmetric 
#' banded matrix A is stored in lower mode
#' if fillmode == 2 then the symmetric banded matrix A is 
#' stored in upper mode
#' @return vector y, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{length of vector y}
#' \item{n: }{1}
#' }
#' @seealso \code{\link{sbmvgpu}} 
#' @export
 

symvgpu <- function(fillmode = 1, alpha = 1, A, x, beta = 0, y)
{
  checkGPU(A)
  checkGPU(x)
  checkGPU(y)
  if (as.integer(A[2]) != as.integer(A[3]))
     stop ("A needs to be square matrix")
  if (as.integer(A[3]) != as.integer(x[2]))
     stop ("A x dimension doesn't match")
  if (as.integer(A[2]) != as.integer(y[2]))
     stop ("A y dimension doesn't match") 
  ext <- .Call(
                "symvGPU",
                 A$ptr,
                 x$ptr,
                 y$ptr,
		   as.numeric(alpha),
                 as.numeric(beta),
                 as.integer(A[2]),
		   as.numeric(fillmode)            
               )
   ext <- GPUobject(ext, as.integer(y[2]), 1)
   return(ext)
}


#' syrgpu
#'
#' This function performs rank 1 update, A = a x x T + A, 
#' where A is symmetric matrix, x is vector, a is scalar
#' by using CUDA cublas function cublasDsyr
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param x input vector; list of R external GPU pointer and dimension
#' @param alpha scale factor a of symmetric banded matrix A; default 1
#' @param fillmode indicates if matrix A lower or upper part is stored, 
#' the other symmetric part is not referenced and is inferred from the 
#' stored elements. if fillmode == 1 then the symmetric 
#' banded matrix A is stored in lower mode
#' if fillmode == 2 then the symmetric banded matrix A is 
#' stored in upper mode
#' @return updated matrix A
#' @seealso \code{\link{gergpu}} 
#' @export
 

syrgpu <- function(fillmode = 1, alpha = 1, x, A)
{
  checkGPU(A)
  checkGPU(x)
  if (as.integer(A[2]) != as.integer(A[3]))
     stop ("A needs to be square matrix")
  if (as.integer(A[3]) != as.integer(x[2]))
     stop ("A x dimension doesn't match")
  ext <- .Call(
                "syrGPU",
                 A$ptr,
                 x$ptr,
		   as.numeric(alpha),
                 as.integer(A[2]),
		   as.numeric(fillmode)            
               )
   ext <- GPUobject(ext, as.integer(A[2]), as.integer(A[3]))
   return(ext)
}


#' syr2gpu
#'
#' This function performs rank 2 update, A = a (x y T + y x T) + A, 
#' where A is symmetric matrix, x is vector, a is scalar
#' by using CUDA cublas function cublasDsyr2
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param x input vector; list of R external GPU pointer and dimension
#' @param y input vector; list of R external GPU pointer and dimension
#' @param alpha scale factor a of symmetric banded matrix A; default 1
#' @param fillmode indicates if matrix A lower or upper part is stored, 
#' the other symmetric part is not referenced and is inferred from the 
#' stored elements. if fillmode == 1 then the symmetric 
#' banded matrix A is stored in lower mode
#' if fillmode == 2 then the symmetric banded matrix A is 
#' stored in upper mode
#' @return updated matrix A
#' @seealso \code{\link{syrgpu}} 
#' @export

syr2gpu <- function(fillmode = 1, alpha = 1, x, y, A)
{
  checkGPU(A)
  checkGPU(x)
  checkGPU(y)
  if (as.integer(A[2]) != as.integer(A[3]))
     stop ("A needs to be square matrix")
  if (as.integer(A[3]) != as.integer(x[2]))
     stop ("A x dimension doesn't match")
  if (as.integer(A[3]) != as.integer(y[2]))
     stop ("A y dimension doesn't match")
  ext <- .Call(
                "syr2GPU",
                 A$ptr,
                 x$ptr,
                 y$ptr,
		   as.numeric(alpha),
                 as.integer(A[2]),
		   as.numeric(fillmode)            
               )
   ext <- GPUobject(ext, as.integer(A[2]), as.integer(A[3]))
   return(ext)
}


#' tbmvgpu
#'
#' This function computes triangular banded matrix-vector multipication x = op(A) x
#' by using CUDA cublas function cublasDtbmv
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param x input/output vector; list of R external GPU pointer and dimension
#' @param k number of sub- or super- diagonals
#' @param trans matrix A transpose operator, 1 (non-transpose), 2 (transpose),
#' 3 (conjugate transpose); default at 1 (non-transpose)
#' @param fillmode indicates if matrix A lower or upper part is stored, 
#' the other symmetric part is not referenced and is inferred from the 
#' stored elements. if fillmode == 1 then the triagular 
#' banded matrix A is stored column by column, with the main diagonal of 
#' the matrix stored in row 1, the first subdiagonal in row 2 
#' (starting at first position), 
#' the second subdiagonal in row 3 (starting at first position), etc. 
#' if fillmode == 2 then the triangular banded matrix A is 
#' stored column by column, with the main diagonal of the matrix stored 
#' in row k+1, the first superdiagonal in row k (starting at second position),
#' the second superdiagonal in row k-1 (starting at third position), etc.
#' @param diagmode indicates whether the main diagonal of the matrix A 
#' is unity and consequently should not be touched or modified by the function.
#' if diagmode = 1, the matrix diagonal has non-unit elements,
#' if diagmode = 2, the matrix diagonal has unit elements
#' @return updated vector x, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{length of vector x}
#' \item{n: }{1}
#' }
#' @seealso \code{\link{gemvgpu}} 
#' @export

tbmvgpu <- function(fillmode = 1, trans = 1, diagmode = 1, k, A, x)
{
  checkGPU(A)
  checkGPU(x)
  if (as.integer(A[2]) != as.integer(A[3]))
     stop ("A needs to be square matrix")
  if (as.integer(A[3]) != as.integer(x[2]))
     stop ("A x dimension doesn't match")
  ext <- .Call(
                "tbmvGPU",
                 A$ptr,
                 x$ptr,
                 as.integer(A[2]),
		   as.integer(k),
		   as.numeric(fillmode),   
		   as.numeric(trans),
		   as.numeric(diagmode)          
               )
   ext <- GPUobject(ext, as.integer(x[2]), 1)
   return(ext)
}


#' tbsvgpu
#'
#' This function solves the triangular banded linear system op(A) x = b
#' by using CUDA cublas function cublasDtbsv
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param x input/output vector; list of R external GPU pointer and dimension
#' @param k number of sub- or super- diagonals
#' @param trans matrix A transpose operator, 1 (non-transpose), 2 (transpose),
#' 3 (conjugate transpose); default at 1 (non-transpose)
#' @param fillmode indicates if matrix A lower or upper part is stored, 
#' the other symmetric part is not referenced and is inferred from the 
#' stored elements. if fillmode == 1 then the triagular 
#' banded matrix A is stored column by column, with the main diagonal of 
#' the matrix stored in row 1, the first subdiagonal in row 2 
#' (starting at first position), 
#' the second subdiagonal in row 3 (starting at first position), etc. 
#' if fillmode == 2 then the triangular banded matrix A is 
#' stored column by column, with the main diagonal of the matrix stored 
#' in row k+1, the first superdiagonal in row k (starting at second position),
#' the second superdiagonal in row k-1 (starting at third position), etc.
#' @param diagmode indicates whether the main diagonal of the matrix A 
#' is unity and consequently should not be touched or modified by the function.
#' if diagmode = 1, the matrix diagonal has non-unit elements,
#' if diagmode = 2, the matrix diagonal has unit elements
#' @return updated vector x, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{length of vector x}
#' \item{n: }{1}
#' }
#' @seealso \code{\link{tbmvgpu}} 
#' @export

tbsvgpu <- function(fillmode = 1, trans = 1, diagmode = 1, k, A, x)
{
  checkGPU(A)
  checkGPU(x)
  if (as.integer(A[2]) != as.integer(A[3]))
     stop ("A needs to be square matrix")
  if (as.integer(A[3]) != as.integer(x[2]))
     stop ("A x dimension doesn't match")
  ext <- .Call(
                "tbsvGPU",
                 A$ptr,
                 x$ptr,
                 as.integer(A[2]),
		   as.integer(k),
		   as.numeric(fillmode),   
		   as.numeric(trans),
		   as.numeric(diagmode)          
               )
   ext <- GPUobject(ext, as.integer(x[2]), 1)
   return(ext)
}


#' trmvgpu
#'
#' This function computes triangular matrix-vector multipication x = op(A) x
#' by using CUDA cublas function cublasDtrmv
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param x input/output vector; list of R external GPU pointer and dimension
#' @param trans matrix A transpose operator, 1 (non-transpose), 2 (transpose),
#' 3 (conjugate transpose); default at 1 (non-transpose)
#' @param fillmode indicates if matrix A lower or upper part is stored, 
#' the other part is not referenced and is inferred from the 
#' stored elements. if fillmode == 1 then the triagular 
#' banded matrix A is stored column by column, with the main diagonal of 
#' the matrix stored in row 1, the first subdiagonal in row 2 
#' (starting at first position), 
#' the second subdiagonal in row 3 (starting at first position), etc. 
#' if fillmode == 2 then the triangular banded matrix A is 
#' stored column by column, with the main diagonal of the matrix stored 
#' in row k+1, the first superdiagonal in row k (starting at second position),
#' the second superdiagonal in row k-1 (starting at third position), etc.
#' @param diagmode indicates whether the main diagonal of the matrix A 
#' is unity and consequently should not be touched or modified by the function.
#' if diagmode = 1, the matrix diagonal has non-unit elements,
#' if diagmode = 2, the matrix diagonal has unit elements
#' @return updated vector x, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{length of vector x}
#' \item{n: }{1}
#' }
#' @seealso \code{\link{gemvgpu}} 
#' @export

trmvgpu <- function(fillmode = 1, trans = 1, diagmode = 1, A, x)
{
  checkGPU(A)
  checkGPU(x)
  if (as.integer(A[2]) != as.integer(A[3]))
     stop ("A needs to be square matrix")
  if (as.integer(A[3]) != as.integer(x[2]))
     stop ("A x dimension doesn't match")
  ext <- .Call(
                "trmvGPU",
                 A$ptr,
                 x$ptr,
                 as.integer(A[2]),
		   as.numeric(fillmode),   
		   as.numeric(trans),
		   as.numeric(diagmode)          
               )
   ext <- GPUobject(ext, as.integer(x[2]), 1)
   return(ext)
}


#' trsvgpu
#'
#' This function solves triangular linear system op(A) x = b
#' by using CUDA cublas function cublasDtrsv
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param x input/output vector; list of R external GPU pointer and dimension
#' @param trans matrix A transpose operator, 1 (non-transpose), 2 (transpose),
#' 3 (conjugate transpose); default at 1 (non-transpose)
#' @param fillmode indicates if matrix A lower or upper part is stored, 
#' the other part is not referenced and is inferred from the 
#' stored elements. if fillmode == 1 then the triagular 
#' banded matrix A is stored column by column, with the main diagonal of 
#' the matrix stored in row 1, the first subdiagonal in row 2 
#' (starting at first position), 
#' the second subdiagonal in row 3 (starting at first position), etc. 
#' if fillmode == 2 then the triangular banded matrix A is 
#' stored column by column, with the main diagonal of the matrix stored 
#' in row k+1, the first superdiagonal in row k (starting at second position),
#' the second superdiagonal in row k-1 (starting at third position), etc.
#' @param diagmode indicates whether the main diagonal of the matrix A 
#' is unity and consequently should not be touched or modified by the function.
#' if diagmode = 1, the matrix diagonal has non-unit elements,
#' if diagmode = 2, the matrix diagonal has unit elements
#' @return updated vector x, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{length of vector x}
#' \item{n: }{1}
#' }
#' @seealso \code{\link{tbsvgpu}} 
#' @export

trsvgpu <- function(fillmode = 1, trans = 1, diagmode = 1, A, x)
{
  checkGPU(A)
  checkGPU(x)
  if (as.integer(A[2]) != as.integer(A[3]))
     stop ("A needs to be square matrix")
  if (as.integer(A[3]) != as.integer(x[2]))
     stop ("A x dimension doesn't match")
  ext <- .Call(
                "trsvGPU",
                 A$ptr,
                 x$ptr,
                 as.integer(A[2]),
		   as.numeric(fillmode),   
		   as.numeric(trans),
		   as.numeric(diagmode)          
               )
   ext <- GPUobject(ext, as.integer(x[2]), 1)
   return(ext)
}


#' addgpu
#'
#' This function computes the element-wise addition of two given 
#' vectors/matrices by using CUDA cublas function cublasDgeam
#' @param x list consisting of R external GPU pointer and dimension 
#' @param y list consisting of R external GPU pointer and dimension
#' @return element-wise addition of two vectors/matrices (x + y), 
#' a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{subtractgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:5
#' a_gpu <- creategpu(a)
#' b_gpu <- creategpu(b)
#' addgpu(a_gpu, b_gpu) -> c_gpu
#' gathergpu(c_gpu)

addgpu <- function(x, y)
{
  checkGPU(x)
  checkGPU(y)
  if (as.integer(x[2]) * as.integer(x[3])
      != as.integer(y[2]) * as.integer(y[3]))
    stop ("vectors dimension don't match")
  ext <- .Call(
                "addGPU",
                x$ptr,
                y$ptr,             
                as.integer(x[2]),
                as.integer(x[3])
               )
  if (as.integer(x[3]) != 1) {
    ext <- GPUobject(ext, as.integer(x[2]), as.integer(x[3]))
  } else {
    ext <- GPUobject(ext, as.integer(y[2]), as.integer(y[3]))
  }
  return(ext)
}


#' subtractgpu
#'
#' This function computes the element-wise subtraction of two
#' given vectors/matrices by using CUDA cublas function cublasDgeam
#' @param x list consisting of R external GPU pointer and dimension 
#' @param y list consisting of R external GPU pointer and dimension
#' @return element-wise subtraction of vectors or matrices (x - y), 
#' a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{addgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:5
#' a_gpu <- creategpu(a)
#' b_gpu <- creategpu(b)
#' subtractgpu(a_gpu, b_gpu) -> c_gpu
#' gathergpu(c_gpu)

subtractgpu<-function(x, y)
{
  checkGPU(x)
  checkGPU(y)
  if (as.integer(x[2]) * as.integer(x[3])
      !=as.integer(y[2]) * as.integer(y[3]))
    stop ("vectors dimension don't match")
  ext <- .Call(
                "subtractGPU",
                x$ptr,
                y$ptr,             
                as.integer(x[2]),
                as.integer(x[3])
              )
  if (as.integer(x[3]) != 1) {
    ext <- GPUobject(ext, as.integer(x[2]), as.integer(x[3]))
  } else {
    ext <- GPUobject(ext, as.integer(y[2]), as.integer(y[3]))
  }
  return(ext)
}


#' multiplygpu
#'
#' This function computes the element-wise multiplication of 
#' two given vectors/matricesby using CUDA cublas function cublasDdgmm
#' @param x list consisting of R external GPU pointer and dimension 
#' @param y list consisting of R external GPU pointer and dimension
#' @return element-wise multiplication of vectors/matrices (x * y), 
#' a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{number of rows}
#' \item{n: }{number of columns}
#' }
#' @seealso \code{\link{dividegpu}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:5
#' a_gpu <- creategpu(a)
#' b_gpu <- creategpu(b)
#' multiplygpu(a_gpu, b_gpu) -> c_gpu
#' gathergpu(c_gpu)

multiplygpu <- function(x, y)
{
  checkGPU(x)
  checkGPU(y)
  if (as.integer(x[2]) * as.integer(x[3])
      != as.integer(y[2]) * as.integer(y[3]))
    stop ("vectors dimension don't match")
  ext <- .Call(
                "vvGPU",
                 x$ptr,
                 y$ptr,
                 as.integer(x[2]) * as.integer(x[3])
              )
  if (as.integer(x[3]) != 1) {
    ext <- GPUobject(ext, as.integer(x[2]), as.integer(x[3]))
  } else {
    ext <- GPUobject(ext, as.integer(y[2]), as.integer(y[3]))
  }
  return(ext)
}


#' mvgpu
#'
#' This function computes the matrix-vector multiplication (X * y) 
#' by using CUDA cublas function cublasDgemv
#' @param X input matrix; list of R external GPU pointer and dimension 
#' @param y input vector; list of R external GPU pointer and dimension
#' @return matrix-vector multiplication (X * y), a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix X's number of rows}
#' \item{n: }{matrix X's number of columns; vector y's number of elements}
#' }
#' @seealso \code{\link{mmgpu}} 
#' @export
#' @examples
#' a <- 1:4
#' b <- 2:3
#' a_gpu <- creategpu(a, 2, 2)
#' b_gpu <- creategpu(b)
#' mvgpu(a_gpu, b_gpu) -> c_gpu
#' gathergpu(c_gpu)

mvgpu <- function(X, y)
{
  checkGPU(X)
  checkGPU(y)
  if (as.integer(X[3]) != as.integer(y[2]))
    stop ("dimension doesn't match")
  ext <- .Call(
                "mvGPU",
                 X$ptr,
                 y$ptr,
                 as.integer(X[2]),
                 as.integer(X[3])            
               )
   ext <- GPUobject(ext, as.integer(X[2]), 1)
   return(ext)
}

