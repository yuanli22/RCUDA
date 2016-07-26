% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/Lapack.R
\name{geamgpu}
\alias{geamgpu}
\title{geamgpu}
\usage{
geamgpu(transa = 1, transb = 1, alpha = 1, A, B, beta = 0, C)
}
\arguments{
\item{A}{input matrix; list of R external GPU pointer and dimension}

\item{B}{input matrix; list of R external GPU pointer and dimension}

\item{C}{output matrix; list of R external GPU pointer and dimension}
}
\value{
updated matrix C, a list consisting of
\itemize{
\item{ptr: }{GPU pointer}
\item{m: }{matrix C's number of rows}
\item{n: }{matrix C's number of columns}
}
}
\description{
This function computes the matrix-matrix addition/trasportation 
C = a op ( A ) + b op ( B ) 
by using CUDA cublas function cublasDgeam
}
\examples{
A_gpu <- creategpu(1:6, 3, 2)
B_gpu <- creategpu(1:6, 3, 2)
C_gpu <- creategpu(1:4, 2, 2)
gemmgpu(2, 1, 1, A_gpu, B_gpu, beta=1, C_gpu)
gathergpu(C_gpu)
}
\seealso{
\code{\link{gemvgpu}}
}
