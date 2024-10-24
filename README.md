java c
MA 575 – LINEAR REGRESSION – FALL 2024
Chapter 1: Introduction and some basic conceptsThe goal of the course is to develop a statistical theory for the linear model, and to learn how to use the linear model in applications. Practical implementation will be done using the statistical software R. The course has lectures and discussions focused mostly on the theoretical aspects, and a lab component focused mostly on practical implementations. Therefore this course assumes familiarity with linear algebra, probability theory and statistical inference (see below for a brief review), and computer programming using R.Lecture notes will be provided at the beginning of each chapter. Most of these notes are taken from the textbook [1]. Another good book that I recommend is [3]. These textbooks are not required, but feel free to get a copy if you wish to read ahead of time, or beyond what we will cover.
This first chapter is a general review of linear algebra, probability and statistics as needed in this class. For more details on these pre-requisites, see textbooks such as [2], and the appendix of [1].
1.  Matrices
A matrix is a collection of real numbers arranged in a rectangular table as below.

I      .         .         .          I
I      .         .         .       I  \ An1     ···   Anm  ,We write A ∈ Rn×m to say that the matrix A hasn rows and m columns. We also say that the row-dimension of A isn, and its column-dimension is m. When m = n we say that A is a square matrix. We write Aij  to denote the number at row iand column j. We will sometimes say that the numbers Aij are the elements of the matrix. For instance, consider the matrix

We see that B ∈ R2×3 . And we have B21  = —1.
If all the elements of A are zero, we say that A is a null matrix, and we write it as 0, or 0n,m if we need to make the dimension explicit. If A is a square matrix, and the only non-zero elements of A are on the diagonal,
we say that A is a diagonal matrix. If in addition all the elements on the diagonal are equal to 1, we say that A is the identity matrix, and we write I or In to make the dimension explicit. For instance I2 denotes the matrix
A matrix with a single column is called a column-vector. A matrix with a single row is called a row-vector. In this class, when we say vector without indicating whether it is a row-vector or a column-vector, we will always mean a column-vector. To refer to a row-vector, we will always explicitly say ’row-vector’ . The number of elements in a row-vector or a column-vector is the size or the dimension of that vector. We will write 0 (resp 1) to denote a vector with all elements equal to 0 (resp. 1). As with matrices, when we want to make the dimension explicit, we will write 0n  and 1n respectively. Given a row-vector or a column-vector u ∈ Rn, its length (or norm) is

1.1. Basic operationsAddition  We are allowed to add two matrices when they have the same number of rows and columns. So, if
A ∈ Rn×m, and B ∈ Rn×m, then A + B is well-defined, it also belongs to Rn×m, and (A + B)ij  = Aij + Bij .
For instance, take
and    .
Then

Multiplication  When the number of columns of A is the same as the number of rows of B, then we can multiply A by B to get AB. Specifically, if A ∈ Rn×m, and B ∈ Rm×k, then AB ∈ Rn×k, and

In other words, (AB)ij is the inner product (see below) of the i-th row of A with the j-th column of B. For instance, take

Then

Note that in general, AB  BA. In fact, ifA ∈ Rn×m, and B ∈ Rm×k, and n  k, then BA is not well-defined.
Transposition  When the rows and columns of a matrix A are interchanged, the resulting matrix is called the transpose of A, and we write it as A′ or sometimes AT . If A′ = A, we say that A is a symmetric matrix. For instance if
then    .
It is easy to check that (A′)′ = A, (A + B)′ = A′ + B′ , (AB)′ = B′A′ . From these rules we see that a matrix of the form. X′X is always symmetric, since (X′X)′  = X′(X′)′ = X′X.Note that the transpose of a column-vector is a row-vector, and vice versa. It is also useful to know that if u ∈ Rn  = Rn×1 is a column-vector, thenu′  ∈ R1×n  is a row-vector, and we can multiply u′ byu according to the multiplication rules given above, and we recover that

Following again the multiplication rules we see that

\ un  ,                         \ unu1     unu2     ···    un(2)   ,
For two column vectors u, v ∈ Rn, the inner product between them is n

Other equivalent notations for u′ v that you may encounter in these notes are〈u, v〉, and uTv. The cosine of the angle between u, vis
And we say that u, v are orthogonal if the cosine of that angle is 0. In other words, if u′ v = 0. If, in addition to be orthogonal,u, v satisfy ⅡuⅡ2  =  ⅡvⅡ2  = 1, we say that u, v are orthonormal. A matrix A ∈ Rn×n  is orthogonal is A′A = In. Note that this is equivalent to saying that the columns of A, viewed as vectors, are orthonormal (meaning that any two of them are orthonormal).
Block matrices  A block matrix (or partitioned matrix) is a matrix of matrices. The addition and multiplication rules for two block matrix with the same block structure are the same as with ordinary matrices, with the difference that we view the blocks as the elements. For instance, we can partition

where
A11 =   , A12 =   ,   A21 =  ,   A33 = 9.
1.2.  Trace
If A ∈ Rn×n, the trace of A is the sum of its diagonal elements:
The following properties are easy to check (please make an effort to check these by yourself): Tr(A′) = Tr(A), If A and B are square matrices with same dimensions, Tr(AB) = Tr(BA), Tr(A+B) = Tr(A)+Tr(B). And for any real number α, Tr(αA) = αTr(A).
Exercise:  If u ∈ Rn is a column-vector, show that

Exercise:  If A ∈ Rm×n show that

It is useful to know that ⅡAⅡF  = √Tr(AAT) defines a norm on matrices known as the Frobenius norm.
1.3. RankGiven vectors A1,..., An  of Rm, a vector of the form. b1A1  + ... + bnAn, for real numbers b1 ,...,bn  is called a linear combination of A1,..., An. The set of all possible linear combinations of A1,..., An is called the subspace (or vector space) spanned (or generated) by A1,..., An. We write this as span(A1,..., An). For instance span(12) is the first diagonal line.Non-zero vectors A1,..., An  are said to be linearly independent if no Ai  can be written as a linear com- bination of the remaining vectors. Equivalently, this means that the only way to get the linear combination b1A1 + ... + bnAn  to be equal to 0m  is to set b1  = ··· = bn  = 0. The dimension of span(A1,..., An) is maximal number of linearly independent columns that we can find among (A1,..., An).Let A  ∈  Rm×n.  The range or column space of A is defined as  Range(A)    {Ab,  b  ∈  Rn }.  Let Aj  denote the j-th column of A. We readily see that Range(A) = span(A1,..., An). This is because the
product Ab ∈ Rm can be written as
The dimension of Range(A) is called the column-rank of A (hence the column-rank of A is the maximum number of linear independent columns that A has). Similarly, the maximal number of linearly independent row we can find in a matrix is called the row-rank of the matrix. An important result in linear algebra says that the column-rank and the row-rank of the matrix are always equal. And we call this common value the rank of the matrix, and we write Rank(A). The following properties are useful to know. Rank(A) = 0 if and only if A = 0. If A ∈ Rm×n , Rank(A) ≤ min(m,n). Rank(A) = Rank(AT). Rank(A) = Rank(AAT) = Rank(AT A).
1.4. InverseIf, for a square matrix A, there exists a matrix B such that AB = BA = I, then we say that A is invertible, and B is its inverse. We write this as B = A−1 . It is not always easy to tell when a square matrix is invertible. And even when we know that a mtrix is invertible, calculating the inverse is not always easy either. However, it is easy to use the definition to check that if A is a diagonal matrix, and all its diagonal elements are non-zero, then A is invertible, and

Also, for 2-by-2 matrices we know how to do. Indeed,

is invertible if and only if ad − bc  0 and

It is often the case that we want to find the inverse of matrices obtained by adding a ”small” perturbation to a matrix A, say. The following result answers that question.
Proposition 1.1 (Woodbury identity).  Let A ∈ Rn×n  known to be invertible, and let U ∈ Rn×q, V ∈ Rq×n. If A + UV is also invertible, then
(A + UV)−1 = A−1 − A−1 U (Iq + VA−1 U)−1 VA−1 .
Exercise:  Establish Proposition 1.1. Simply multiply A+UV by A−1 −A−1 U (Iq + VA−1 U)−1 VA−1 and show that it is equal to In.
1.5.  Subspaces and ProjectorsLet U1,..., Un   ∈  Rm,  and consider the matrix U  =  [U1,..., Un], that is, the matrix  U with  columns U1,..., Un. Given the subspace Range(U) = span(U1,..., Un) that we denote U for simplicity, and given a vector x  ∈ Rm, consider the problem of finding the element ˆ(x) of U that is closest to x. We call ˆ(x) the orthogonal projection of x on U. We can write this in equations as follows: find ˆ(x) such thatWhen Rank(U) = n (which means that m ≥ n), then the solution to (1) is   ˆ(x) = Px,    where    P = U(U⊤ U)−1 U⊤ .P is called the orthogonal projector on span(U1,..., Un). The proof of this result goes as follows. First note that P⊤ = P, and P2  = U(U⊤ U)−1 U⊤ U(U⊤ U)−1 U⊤ = P. Furthermore, if z ∈ U, thenz must be of the form. z = Ub for some b ∈ Rn. Therefore, Pz = U(U⊤ U)−1 U⊤ z = U(U⊤ U)−1 U⊤ Ub = Ub = z. And for all x ∈ Rm , z⊤ (I −P)x = (Pz)⊤ (I −P)x = z⊤P⊤x−z⊤P⊤Px = z⊤Px−z⊤Px = 0. Now, let’s return to (1). The square of the left-hand side is ∥x − Px∥2(2). The square of the right-hand side is for any z ∈ U is∥Px+(x−Px)−z∥2(2) = ∥Px−z∥2(2)+∥x−Px∥2(2)+2(Px−z)⊤ (x−Px) = ∥Px−z∥2(2)+∥x−Px∥2(2) ≥ ∥x−Px∥2(2) ,
where we use the fact Px − z ∈ U.More generally, a projector is any symmetric matrix P  ∈ Rm×m  such that P2  = P. Some textbooks use the terminology idempotent matrices. This definition does not immediately say what sub-space P is projecting on. However, this often is easy to find. It is given by {Px,x ∈ Rm }.
Example 1. Let 1n = (1, . . . , 1)T  ∈ Rn be the column-vector with all components equal to 1. And let

Clearly, MT = M, and
Recall that 1n(T)1n  = n. Hence 1n1n(T)1n1n(T) = n1n1n(T) . It follows that M2  = M. Hence M is a projector. What is
M projecting on? We note that for any b ∈ Rn ,

Hence, M is projecting on the orthogonal to 1n. In statistical terms, M transforms b into a vector that is centered.
1.6. Eigen-decomposition and quadratic forms
Let A ∈ Rn×n  a square matrix. A non-zero vector u ∈ Rn  is an eigenvector for A if there exists λ ∈ R such that
Au = λu.
We say that the matrix A is diagonalizable if we can find n vectors u1 , . . . , un  such that any two of them are
orthonormal (uj(T)uk  = 0 for j  k, and ∥uj∥2  = ∥uk∥2  = 1), and λ1 ,...,λn  ∈ R such that
Auj  = λjuj ,   1 ≤ j ≤ n.
If we define U ∈ Rn×n with j-th column uj, and w代 写MA 575 – LINEAR REGRESSION – FALL 2024 Chapter 1: Introduction and some basic conceptsPython
代做程序编程语言e let Λ be the diagonal matrix with j-th diagonal element λj, then the definition above can be written equivalently as
AU = UΛ .The fact that the vectors u1 , . . . , un  are orthonormal is equivalent to UT U  = In. That is to say, the ma- trix U is invertible and its inverse is U −1  = U′ . Such matrices are called orthogonal matrices. Hence A is diagonalizable if there exist a diagonal matrix Λ and an orthogonal matrix U such thatA = UΛUT .                                                                   (2)
The equation in (2) is called the eigen-decomposition of A. Not all matrices are diagonalizable. An important result of linear algebra says that if A is symmetric (meaning AT  = A), then A is diagonalizable.Suppose now that A ∈ Rn×n  is a symmetric matrix. The function that takes a vector x ∈ Rn, and returns the number xTAx is called a quadratic form. Following the matrix multiplication rules given above we can work our the expanded form. of xTAx as
The matrix A is said to be semi-positive definite if xTAx ≥ 0 for all x ∈ Rn. If xTAx > 0 for all x  0 ∈ Rn , we say that A is positive definite. An important result of linear algebra says that a symmetric matrix A is semi- positive definite if and only if all its eigenvalues are nonnegative. And it is positive definite if all its eigenvalues are positive.We will use this concept to compare two symmetric matrices. So, if A, B  ∈  Rn×n  are two symmetric matrices, we say that A is greater or equal to B (or no smaller than), and we write A  ⪰  B, if A − B is semi-positive definite. In other words, if for all x ∈ Rn , xTAx ≥ xT Bx. In other words if all the eigenvalues of A − B are nonnegative.
2.  Random variablesConsider a sample space Ω with a probability P. A random variable is a function X  :   Ω  →  R. Random variables are typically obtained when we take measurements after a random experiment. For instance, consider the experiment of randomly selecting an individual from a population, and randomly selecting a location on planet Earth. In the first case, for instance measuring the age of that individual gives a random variable. In the second example, measuring the average temperature at that location gives us a random variable.Remark 1. A general convention in probability theory and statistics is to denote random variables using capital letters (X,Y, Z), and to use lower case to denote their realizations, or the possible values they can assume. Hence P(X  = x) refers to the probability that the random variable X takes value x. However for various reasons, we usually do not follow this convention in statistical modeling. Hence in this class, although we will make a distinction between capital and lower case letters, we will not systematically use capital letters to denote random variables.
If X :  Ω → R is a random variable, its cumulative distribution function, or cdf, is F(x) = P(X ≤ x),  x ∈ R.
If X can take only a countable (discrete) number of values {x1 ,... xn,...}, say, then it has a probability mass function, or pmf
pi = P(X = xi), i ≥ 1.
If X can take value in a continuum, and has zero probability of taking a specific value, then it has a probability density function, or pdf f, and
P(X ∈ A) = f(x)dx.
If X  :   Ω → R is a random variable, we typically summarize the location around which X typically takes values by calculating its expected value or mean E(X)
xipi ,  provided    |xi|pi  < 
If Σi|xi|pi = ∞ we say that E(X) does not exists. In the continuous case
E(X) = ∫ xf(x)dx,   provided   ∫ |x|f(x)dx < ∞ . If ∫ |x|f(x)dx = ∞ , we say that E(X) does not exists.
We typically summarize the square of the fluctuation around its mean (when it exists) by the variance Var(X). If X is discrete with pmfp:

If X is continuous, with pdf f:

More generally, if his a real-valued function,
and in the continuous case          E(h(X))    =    ∫ h(x)f(x)dx.
In particular the variance can also be written as
Var(X) = E(X2) − (E(X))2 . Notice that, since Var(X) ≥ 0, we have:
(E(X))2  ≤ E(X2).
Here is an important properties of the expectation and the variance. Let X be a random variable. For any numbers a,b,
E(aX + b) = aE(X) + b,   Var(aX + b) = a2Var(X).
2.1. Random vectorsGiven a random experiment with sample space Ω, a multivariate random variable (or a random vector) is a function X  :   Ω  →  Rp. In other words, we perform. an experiment, and take several measurements. For instance, we randomly pick a municipality in the country, and we measure the population, the proportion ofresidents older than 65, and the average summer temperature of that municipality. We have a 3-dimensional random vector. To save time we focus on continuous random vectors. They are characterized by a joint pdf f such that for A ⊆ Rp ,
P(X ∈ A) =  f(x)dx,
where the integral given above is a multiple integral.
If X is a random vector X  :  Ω →  Rp, we summarize its distribution using its expected vector or mean, and its covariance matrix. In the continuous case (with density f) the expected vector is

The covariance matrix summarizes all the joint fluctuations:

\ Cov(Xp, X1 )   ...   Cov(Xp, Xp) ,
where

Notice that in the case of a random vector, Var(X) is a matrix (the covariance matrix). More generally, if h :  Rp  → Rq, with h(x) = (h1 (x),..., hq(x)), the expectation of h(X) is defined as

Here are two important examples of function h. Let b ∈ Rp  a vector, and A ∈ Rq×p  a matrix. Then bTX and AX are well-defined transformations of X.
Exercise:  Let µ ∈  Rp denote the expectation of X , and Σ its covariance matrix. For any b ∈ Rp, show that E(bTX) = bT µ, and Var(bTX) = bT Σb.
Exercise:  For any A ∈ Rq×p, show that E(AX) = Aµ, and Var(AX) = AΣAT .
Let  (X, Y)  ∈   (Rp , Rq ) be  two random vectors with joint pdf fX,Y  (this is a density on Rp+q). The marginal pdf of X is a
fX (x) =  fX,Y(x,y)dy.
fX is a density on Rp. The marginal density fY  of Y is defined similarly. The conditional density of X given Y = y is

The two random vectors X and Y are independent if
fX|Y(x|y) = fX (x),  for all x ∈ Rp, y ∈ Rq .This is equivalent to fY |X (y|x) = fY (y), and to fX,Y(x,y) = fX (x)fY (y), for all x ∈ Rp,  y ∈ Rq. Be careful with the definition: for instance, if X = (X1, X2 ) and Y = X3 , X and Y independent is not the same as X1, X2 , X3 independent.
The conditional expected vector of X given Y = y is defined as the vector
∈ Rp.
More generally, if h : Rp → Rd, the conditional expected vector of h(X) given Y = y is defined as the vector

2.2.  On the multivariate Gaussian distributionp random variables taken jointly as X  =  (X1,..., Xp)′   ∈  Rp  form. a multivariate normal (or Gaussian) random vector if for all vectors a ∈ Rp, the random variable a′X is a normal random variable. Remarkably, the distribution of a multivariate normal random variable X is completely characterized by its expected value µ, and covariance matrix Σ, where

1.  If (X1, X2 )′  ∈ R2 is a two-dimensional Gaussian random vector, then X1  and X2  are independent if and only if Cov(X1, X2 ) = 0.
2.  If X = (X1,..., Xp)′  ∼ Np(µ,Σ), and a ∈ Rp, then a′X ∼ N(a′µ,a′Σa).
3.  If X = (X1,..., Xp)′  ∼ Np(µ,Σ), and A ∈ Rk×p, for some k ≥ 1, then AX ∼ N(Aµ, AΣA′).Example 2.  Suppose Xi  i .  N(0,σi(2)). Then jointly, X = (X1,..., Xp)′  ∼ Np(0, Σ), where Σ is a diagonal
matrix,with i-th diagonal given by σi(2) . Therefore, as a consequence of property (2) above, for any vector a ∈ Rp , a′X ∼ N(0, Σi σi(2)ai(2)).
A list of important probability distributions that all statisticians are expected to know:
•  discrete: Bernoulli, Binomial, Poisson, Geometric.
•  continuous: Exponential, Gamma, Inverse-gamma, Gaussian.
•  Multivariate: Multivariate normal.
3.  Statistical inferenceIn statistics a variable is a numerical characteristic or construct, defined on some population, and that can be measured. In this class we will typically view statistical variables as random variables. For instance, the height of an individual in a population is a variable. But for the purpose of statistical modeling, we will view height as the (random) number we get by randomly selecting one individual from the population and measuring their height. Statistical inference is concerned with learning population distributions from finite sample data. A lot of different scientific communities deal with data. What is unique to statistics and machine learning is that we almost systematically build statistical models to deal with data. This has the benefit of giving us a representation of the data generating process that we can use for many other purposes.Hence, given data x1 , . . . , xn, we start by making the assumption that these are realization of random variables X1,..., Xn, for which we formulate a statistical model: that is, we postulate a family of probability laws {fθ,  θ  ∈   Θ}, with the assumption that one of them (θ⋆   say) has generated the data. Typically we focus on densities. We then use the data to estimate θ⋆ . This can be done in many many ways: maximum likelihood, least squares, method of moments, and many other inferential procedures. In this class, we use almost exclusively the least squares, but it is important to remember maximum likelihood.
3.1. Maximum likelihood estimationSuppose that we observe a random sample X1,..., Xn  iid from some unknown density f. We postulate a statistical model {fθ,  θ  ∈  Θ}  and assume that f  = fθ*   for some (unknown) value θ⋆    ∈  Θ . Let us write X1,..., Xn for the random variables and x1 , . . . , xn for the observed data. We would like to use the observed
data to infer θ⋆. If the data were generated from fθ, the joint density of the random vector (X1,..., Xn) is the function

One possible way to estimate θ⋆  is to calculate fθ(x1 ,..., xn) for each value of θ and see which value of θ gives the highest density. It then makes sense to use this value as an estimate for θ⋆ .
The function θ →I fθ(x1 ,..., xn) is called the likelihood function and denote L:

The logofL is called log-likelihood function:
The value of θ that maximizes L(θ|x1 ,..., xn) is called the maximum likelihood estimator (MLE). Clearly, maximizing L is equivalent to maximizing ℓ but for numerical stability we will often prefer working with the latter. It is important to note that in the likelihood, x1 , . . . , xn  represents the observed data.Example 3.  Suppose that  we observe a sample  (x1 ,..., xn) that we believe to be generated iid-ly from N(µ,1), where the mean µ is unknown and is to be estimated. The density of the model is exp(−(x − µ)2 /2)/√2π . Therefore the likelihood function is

The log-likelihood is ℓ(µ) = −0.5nlog(2π) − 0.5 Σ(xi  − µ)2 . ℓ is a parabola. Taking the derivative wrt µ, we see that the value of µ that maximize ℓ(µ) is

the sample mean. Thus the sample mean is also themle estimator of µ in this model.



         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
