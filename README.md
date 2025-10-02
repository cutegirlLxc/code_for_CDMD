The following is the code to implement CDMD.

```r

library(Matrix)
library(skmeans)
library(parallel)
library(foreach)
library(doParallel)
library(cluster)
library(skmeans)



error <- function(true, data) {

  recode_actual_labels <- as.numeric(factor(true, levels = unique(true)))
  recode_kmeans_results <- as.numeric(factor(data, levels = unique(data)))
  
  correct_samples <- sum(recode_actual_labels == recode_kmeans_results)
  
  accuracy <- correct_samples / length(true)
  return(1 - accuracy)
}


pi <- function(n, ky, kz, community_sizes, dmax, dmin, dmax1, dmin1) {
  
  
  pi0 <- matrix(0, nrow = n, ncol = ky)
  pi1 <- matrix(0, nrow = n, ncol = kz)
  size2 <- n / kz
  
  start_index <- 1
  for (i in 1:ky) {
    csize <- community_sizes[i]
    
    
    part_size <- csize / 4
    
    if (csize %% 4 != 0) {
      stop("Community sizes must be divisible by 4.")
    }
    
    end_index <- start_index + part_size - 1
    pi0[start_index:end_index, i] <- rep(dmax, part_size)  # 第一部分为dmax
    start_index <- end_index + 1
    end_index <- start_index + part_size - 1
    pi0[start_index:end_index, i] <- rep(dmin, part_size)  # 第二部分为dmin
    start_index <- end_index + 1
    end_index <- start_index + part_size - 1
    pi0[start_index:end_index, i] <- rep(dmax1, part_size)  # 第三部分为dmax1
    start_index <- end_index + 1
    end_index <- start_index + part_size - 1
    pi0[start_index:end_index, i] <- rep(dmin1, part_size)  # 第四部分为dmin1
    start_index <- end_index + 1
  }
  
  for (i in 1:kz) {
    ind1 <- (i-1) * size2 + 1
    ind2 <- i * size2
    pi1[ind1:ind2, i] <- 1
  }
  
  return(list(pi0 = pi0, pi1 = pi1))
}


pi <- function(n, ky, kz, community_sizes,dmax,dmin,dmax1,dmin2) {
 
  
  pi0 <- matrix(0, nrow = n, ncol = ky)
  pi1 <- matrix(0, nrow = n, ncol = kz)
  size2 <- n / kz
  
  start_index <- 1
  for (i in 1:ky) {
    csize<-community_sizes[i]
    end_index <- start_index + community_sizes[i] - 1
    pi0[start_index:end_index, i] <- c(rep(dmax, csize/2), rep(dmin, csize/2))
    start_index <- end_index + 1
  }
  
  for (i in 1:kz) {
    ind1 <- (i-1) * size2 + 1
    ind2 <- i * size2
    pi1[ind1:ind2, i] <- 1
  }
  
  return(list(pi0 = pi0, pi1 = pi1))
}

adja <- function(ky,kz, n, rho,pi0,pi1) {
  b <- matrix(runif(ky*kz), ncol = kz)
  pm <- rho* (pi0 %*% b %*% t(pi1))
  adj <- matrix(rbinom(n*n, 1, pm), nrow = n, ncol = n)
  return(adj)
}  




truth <- function(ni, k) {
  tru <- unlist(mapply(rep, x = 1:k, times = ni))
  return(tru)
}


row_norm<-function(x){
  normx<-norm(x,type = "2")
  x/normx
}



adja_sum<-function(n,L,ky,kz,rho,pi0,pi1){
  AA<-matrix(0,n,n)
  for (i in 1:L) {
    aj<-adja(ky,kz,n,rho,pi0,pi1)
    AA<-AA+aj%*%t(aj)
  }
  diag(AA)<-0
  return(AA)
}



# An example


set.seed(54321)
n<-600
ky <- 3
kz <- 2

ni<-c(100,200,300)
rho <- 0.03
Pi<-pi(n,ky,kz,ni,1.8,0.2,1.5,0.5)
pi0<-Pi[[1]]
pi1<-Pi[[2]]
truelabel<-truth(ni,ky)


L <- 15
e<-numeric(50)

for (j in 1:50) {
  S<-adja_sum(n,L,ky,kz,rho,pi0,pi1)
  U<-as.matrix(eigen(S)$vectors[,1:ky])
  U_star<-t(apply(U, MARGIN = 1,row_norm))
  df_y<-data.frame(U_star)
  cl_y <- kmeans(df_y,ky)$cluster
  e[j]<-error(truelabel,cl_y)
}

error_rate <-mean(e)



# estimate k_y

p_hat<-function(matrix_list){
  matrix_means <- sapply(matrix_list, mean)
  rho_hat<-mean(matrix_means)
  return(rho_hat)
}



compute_tn <- function(n, L, p_hat) {
  log_n_plus_L <- log(n + L)
  condition <- p_hat <= sqrt(log_n_plus_L / L)
  
  if (condition) {
    tn <- (sqrt(L) * n * p_hat)^(5/4) * log(n + L)^(3/8)
  } else {
    tn <- L * n * p_hat^2 * sqrt(log(n))
  }
  
  return(tn)
}








```
