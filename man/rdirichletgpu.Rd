% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/random_number_genertor.R
\name{rdirichletgpu}
\alias{rdirichletgpu}
\title{rdirichletgpu}
\usage{
rdirichletgpu(n, alpha, seed = 1)
}
\arguments{
\item{n}{number of random numbers}

\item{alpha}{concentration parameters of Dirichlet distribution;}

\item{seed}{random number generator seed; default value 1}
}
\value{
generated random numbers vector, a list consisting of
\itemize{
\item{ptr: }{GPU pointer}
\item{m: }{number of rows}
\item{n: }{number of columns}
}
}
\description{
This function generates Dirichlet distributed random numbers 
by using self-defined CUDA function based on George Marsaglia 
and Wai Wan Tsang's method and gamma/Dirichlet relationship
}
\examples{
a_gpu <- rdirichletgpu(100, 2, 1) 
}
\seealso{
\code{\link{runifgpu}}
}

