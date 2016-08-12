#' gemmgpu
#'
#' This function computes the matrix-matrix multiplication 
#' C = a op ( A ) op ( B ) + b C 
#' by using CUDA cublas function cublasDgemm
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param B input matrix; list of R external GPU pointer and dimension
#' @param C input/output matrix; list of R external GPU pointer and dimension
#' @param alpha scale factor a of matrix A; default 1
#' @param beta scale factor b of matrix C; default 0
#' @param transa matrix A transpose operator, 1 (non-transpose), 2 (transpose),
#' 3 (conjugate transpose); default at 1 (non-transpose)
#' @param transb matrix B transpose operator, 1 (non-transpose), 2 (transpose),
#' 3 (conjugate transpose); default at 1 (non-transpose)
#' @return updated matrix C, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix C's number of rows}
#' \item{n: }{matrix C's number of columns}
#' }
#' @seealso \code{\link{gemvgpu}}  
#' @export
#' @examples
#' A_gpu <- creategpu(1:6, 3, 2)
#' B_gpu <- creategpu(1:6, 3, 2)
#' C_gpu <- creategpu(1:4, 2, 2)
#' gemmgpu(2, 1, 1, A_gpu, B_gpu, beta=1, C_gpu)
#' gathergpu(C_gpu)

gemmgpu <- function(transa = 1, transb = 1, alpha = 1, A, B, beta = 0, C)
{
  checkGPU(A)
  checkGPU(B)
  checkGPU(C)
  if ((transa != 1) && (transa != 2) && (transa != 3))
    stop ("A transpose operation input error")
  if ((transb != 1) && (transb != 2) && (transb != 3))
    stop ("B transpose operation input error")
  if (!is.numeric(beta) || !is.numeric(alpha))
    stop ("scale factor should be numerical")
  if ((transa == 1) && (transb == 1)) {
    if (as.integer(A[2]) != as.integer(C[2]))
      stop ("A and C dimensions not match")
    if (as.integer(A[3]) != as.integer(B[2]))
      stop ("A and B dimensions not match")
    if (as.integer(B[3]) != as.integer(C[3]))
      stop ("B and C dimensions not match")
    m <- as.integer(A[2])
    n <- as.integer(B[3])
    k <- as.integer(A[3])
    }
  if ((transa != 1) && (transb == 1)) {
    if (as.integer(A[3]) != as.integer(C[2]))
      stop ("A and C dimensions not match")
    if (as.integer(A[2]) != as.integer(B[2]))
      stop ("A and B dimensions not match")
    if (as.integer(B[3]) != as.integer(C[3]))
      stop ("B and C dimensions not match")
    m <- as.integer(A[3])
    n <- as.integer(B[3])
    k <- as.integer(A[2])
    }
  if ((transa == 1) && (transb != 1)) {
    if (as.integer(A[2]) != as.integer(C[2]))
      stop ("A and C dimensions not match")
    if (as.integer(A[3]) != as.integer(B[3]))
      stop ("A and B dimensions not match")
    if (as.integer(B[2]) != as.integer(C[3]))
      stop ("B and C dimensions not match")
    m <- as.integer(A[2])
    n <- as.integer(B[2])
    k <- as.integer(A[3])
    }
  if ((transa != 1) && (transb != 1)) {
    if (as.integer(A[3]) != as.integer(C[2]))
      stop ("A and C dimensions not match")
    if (as.integer(A[2]) != as.integer(B[3]))
      stop ("A and B dimensions not match")
    if (as.integer(B[2]) != as.integer(C[3]))
      stop ("B and C dimensions not match")
    m <- as.integer(A[3])
    n <- as.integer(B[2])
    k <- as.integer(A[2])
    }
  ext <- .Call(
                "gemmGPU",
                 A$ptr,
                 B$ptr,
                 C$ptr,
                 as.integer(A[2]),
                 as.integer(B[2]),
                 as.integer(C[2]),	
                 m,
                 n,
                 k,	
                 as.numeric(transa),
                 as.numeric(transb),
                 as.numeric(alpha),
                 as.numeric(beta)
              )
   ext <- GPUobject(ext, as.integer(C[2]), as.integer(C[3]))
   return(ext)
}


#' symmgpu
#'
#' This function computes the symmetric matrix-matrix multiplication 
#' C = a A B + b C 
#' by using CUDA cublas function cublasDsymm
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param B input matrix; list of R external GPU pointer and dimension
#' @param C input/output matrix; list of R external GPU pointer and dimension
#' @param alpha scale factor a of matrix AB; default 1
#' @param beta scale factor b of matrix C; default 0
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
#' @param sidemode indicates whether the given matrix is on the left or right side
#' in the matrix equation solved by a particular function. If sidemode == 1, 
#' the matrix is on the left side in the equation If sidemode == 2, 
#' the matrix is on the right side in the equation.
#' @return updated matrix C, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix C's number of rows}
#' \item{n: }{matrix C's number of columns}
#' }
#' @seealso \code{\link{gemmgpu}}  
#' @export
 

symmgpu <- function(sidemode = 1, fillmode = 1, alpha = 1, A, B, beta = 0, C)
{
  checkGPU(A)
  checkGPU(B)
  checkGPU(C)
  if (as.integer(A[2]) != as.integer(A[3]))
    stop ("A should be square matrix")
  if (as.integer(A[3]) != as.integer(B[2]))
    stop ("A and B dimensions not match")
  if (as.integer(A[3]) != as.integer(C[2]))
    stop ("A and C dimensions not match")
  if (as.integer(B[3]) != as.integer(C[3]))
    stop ("B and C dimensions not match")
  if (!is.numeric(beta) || !is.numeric(alpha))
    stop ("scale factor should be numerical")
  ext <- .Call(
                "symmGPU",
                 A$ptr,
                 B$ptr,
                 C$ptr,
                 as.integer(A[2]),
                 as.integer(B[2]),
                 as.integer(C[2]),	
                 as.numeric(fillmode),	
                 as.numeric(sidemode),	
                 as.integer(A[2]),
                 as.integer(C[3]),	
                 as.numeric(alpha),
                 as.numeric(beta)
              )
   ext <- GPUobject(ext, as.integer(C[2]), as.integer(C[3]))
   return(ext)
}


#' syrkgpu
#'
#' This function performs the symmetric rank- k update 
#' C = a op ( A ) op ( A ) T + b C
#' by using CUDA cublas function cublasDsyrk
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param C input/output matrix; list of R external GPU pointer and dimension
#' @param alpha scale factor a; default 1
#' @param beta scale factor b; default 0
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
#' @return updated matrix C, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix C's number of rows}
#' \item{n: }{matrix C's number of columns}
#' }
#' @seealso \code{\link{gemmgpu}}  
#' @export
 

syrkgpu <- function(fillmode = 1, trans = 1, alpha = 1, A, beta = 0, C)
{
  checkGPU(A)
  checkGPU(C)
  if (trans == 1) {
    if (as.integer(A[2]) != as.integer(C[2]))
      stop ("A and C dimensions not match")
    n <- as.integer(A[2])
    k <- as.integer(A[3])  
  }
  if (trans != 1) {
    if (as.integer(A[3]) != as.integer(C[2]))
      stop ("A and C dimensions not match")
    n <- as.integer(A[3])
    k <- as.integer(A[2])  
  }
  if (!is.numeric(beta) || !is.numeric(alpha))
    stop ("scale factor should be numerical")
  ext <- .Call(
                "syrkGPU",
                 A$ptr,
                 C$ptr,
                 as.integer(A[2]),
                 as.integer(C[2]),	
                 as.numeric(fillmode),	
                 as.numeric(trans),	
                 n,
                 k,	
                 as.numeric(alpha),
                 as.numeric(beta)
              )
   ext <- GPUobject(ext, as.integer(C[2]), as.integer(C[3]))
   return(ext)
}


#' syr2kgpu
#'
#' This function performs the symmetric rank- 2k update 
#' C = a(op ( A )op ( B ) T + op ( B )op ( A )T)  + b C
#' by using CUDA cublas function cublasDsyr2k
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param B input matrix; list of R external GPU pointer and dimension 
#' @param C input/output matrix; list of R external GPU pointer and dimension
#' @param alpha scale factor a ; default 1
#' @param beta scale factor b; default 0
#' @param trans matrix A and B transpose operator, 1 (non-transpose), 2 (transpose),
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
#' @return updated matrix C, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix C's number of rows}
#' \item{n: }{matrix C's number of columns}
#' }
#' @seealso \code{\link{syrkgpu}}  
#' @export
 

syr2kgpu <- function(fillmode = 1, trans = 1, alpha = 1, A, B, beta = 0, C)
{
  checkGPU(A)
  checkGPU(B)
  checkGPU(C)
  if ((as.integer(A[2]) != as.integer(B[2]))||
       (as.integer(A[3]) != as.integer(B[3])))
      stop ("A and B dimensions not match")
  if (trans == 1) {
    if (as.integer(A[2]) != as.integer(C[2]))
      stop ("A and C dimensions not match")
    n <- as.integer(A[2])
    k <- as.integer(A[3])  
  }
  if (trans != 1) {
    if (as.integer(A[3]) != as.integer(C[2]))
      stop ("A and C dimensions not match")
    n <- as.integer(A[3])
    k <- as.integer(A[2])  
  }
  if (!is.numeric(beta) || !is.numeric(alpha))
    stop ("scale factor should be numerical")
  ext <- .Call(
                "syr2kGPU",
                 A$ptr,
                 B$ptr,
                 C$ptr,
                 as.integer(A[2]),
                 as.integer(A[2]),
                 as.integer(C[2]),	
                 as.numeric(fillmode),	
                 as.numeric(trans),	
                 n,
                 k,	
                 as.numeric(alpha),
                 as.numeric(beta)
              )
   ext <- GPUobject(ext, as.integer(C[2]), as.integer(C[3]))
   return(ext)
}


#' trmmgpu
#'
#' This function computes the triangle matrix-matrix multiplication 
#' C = a A B or C = a B A 
#' by using CUDA cublas function cublasDtrmm
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param B input matrix; list of R external GPU pointer and dimension
#' @param C input/output matrix; list of R external GPU pointer and dimension
#' @param alpha scale factor a of matrix AB; default 1
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
#' @param trans matrix A transpose operator, 1 (non-transpose), 2 (transpose),
#' 3 (conjugate transpose); default at 1 (non-transpose)
#' @param diagmode indicates whether the main diagonal of the matrix A 
#' is unity and consequently should not be touched or modified by the function.
#' if diagmode = 1, the matrix diagonal has non-unit elements,
#' if diagmode = 2, the matrix diagonal has unit elements.
#' @param sidemode indicates whether the given matrix is on the left or right side
#' in the matrix equation solved by a particular function. If sidemode == 1, 
#' the matrix is on the left side in the equation If sidemode == 2, 
#' the matrix is on the right side in the equation.
#' @return updated matrix C, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix C's number of rows}
#' \item{n: }{matrix C's number of columns}
#' }
#' @seealso \code{\link{symmgpu}}  
#' @export
 

trmmgpu <- function(sidemode = 1, fillmode = 1, trans = 1,
                    diagmode = 1, alpha = 1, A, B, C)
{
  checkGPU(A)
  checkGPU(B)
  checkGPU(C)
  if (as.integer(A[2]) != as.integer(A[3]))
    stop ("A should be square matrix")
  if (as.integer(A[3]) != as.integer(B[2]))
    stop ("A and B dimensions not match")
  if (as.integer(A[3]) != as.integer(C[2]))
    stop ("A and C dimensions not match")
  if (as.integer(B[3]) != as.integer(C[3]))
    stop ("B and C dimensions not match")
  if (!is.numeric(alpha))
    stop ("scale factor should be numerical")
  ext <- .Call(
                "trmmGPU",
                 A$ptr,
                 B$ptr,
                 C$ptr,
                 as.integer(A[2]),
                 as.integer(B[2]),
                 as.integer(C[2]),	
                 as.numeric(trans),
                 as.numeric(fillmode),
                 as.numeric(diagmode),	
                 as.numeric(sidemode),	
                 as.integer(A[2]),
                 as.integer(C[3]),	
                 as.numeric(alpha)
              )
   ext <- GPUobject(ext, as.integer(C[2]), as.integer(C[3]))
   return(ext)
}


#' trsmgpu
#'
#' This function solves the triangle linear system 
#' A X = a B or X A = a B 
#' by using CUDA cublas function cublasDtrsm
#' @param A input matrix; list of R external GPU pointer and dimension 
#' @param B input/output matrix; list of R external GPU pointer and dimension
#' @param alpha scale factor a; default 1
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
#' @param trans matrix A transpose operator, 1 (non-transpose), 2 (transpose),
#' 3 (conjugate transpose); default at 1 (non-transpose)
#' @param diagmode indicates whether the main diagonal of the matrix A 
#' is unity and consequently should not be touched or modified by the function.
#' if diagmode = 1, the matrix diagonal has non-unit elements,
#' if diagmode = 2, the matrix diagonal has unit elements.
#' @param sidemode indicates whether the given matrix is on the left or right side
#' in the matrix equation solved by a particular function. If sidemode == 1, 
#' the matrix is on the left side in the equation If sidemode == 2, 
#' the matrix is on the right side in the equation.
#' @return updated matrix B, a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix B's number of rows}
#' \item{n: }{matrix B's number of columns}
#' }
#' @seealso \code{\link{trmmgpu}}  
#' @export
 

trsmgpu <- function(sidemode = 1, fillmode = 1, trans = 1,
                    diagmode = 1, alpha = 1, A, B)
{
  checkGPU(A)
  checkGPU(B)

  if (as.integer(A[2]) != as.integer(A[3]))
    stop ("A should be square matrix")
  if (as.integer(A[3]) != as.integer(B[2]))
    stop ("A and B dimensions not match")
  if (!is.numeric(alpha))
    stop ("scale factor should be numerical")
  ext <- .Call(
                "trsmGPU",
                 A$ptr,
                 B$ptr,
                 as.integer(A[2]),
                 as.integer(B[2]),
                 as.numeric(trans),
                 as.numeric(fillmode),
                 as.numeric(diagmode),	
                 as.numeric(sidemode),	
                 as.integer(A[2]),
                 as.integer(B[3]),	
                 as.numeric(alpha)
              )
   ext <- GPUobject(ext, as.integer(B[2]), as.integer(B[3]))
   return(ext)
}


#' mmgpu
#'
#' This function computes the matrix-matrix multiplication (X * Y) 
#' by using CUDA cublas function cublasDgemm
#' @param X input matrix; list of R external GPU pointer and dimension 
#' @param Y input matrix; list of R external GPU pointer and dimension
#' @return matrix-matrix multiplication (X * Y), a list consisting of
#' \itemize{
#' \item{ptr: }{GPU pointer}
#' \item{m: }{matrix X's number of rows}
#' \item{n: }{matrix Y's number of columns}
#' }
#' @seealso \code{\link{mmgpu}}  
#' @export
#' @examples
#' a <- 1:6
#' b <- 2:7
#' a_gpu <- creategpu(a, 2, 3)
#' b_gpu <- creategpu(b, 3, 2)
#' mmgpu(a_gpu, b_gpu) -> c_gpu
#' gathergpu(c_gpu)

mmgpu <- function(X, Y)
{
  checkGPU(X)
  checkGPU(Y)
  if (as.integer(X[3]) != as.integer(Y[2]))
    stop ("dimension doesn't match")
  ext <- .Call(
                "mmGPU",
                 X$ptr,
                 Y$ptr,
                 as.integer(X[2]),
                 as.integer(Y[3]),
                 as.integer(X[3])            
              )
   ext <- GPUobject(ext, as.integer(X[2]), as.integer(Y[3]))
   return(ext)
}