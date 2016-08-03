# Asymmetric Multi-Task Learning(AMTL)
* Giwoong Lee(UNIST), Eunho Yang(KAIST), Sung Ju Hwang(UNIST)

Asymmetric Multi-Task Learning code, If you want to use it, please let me know and cite AMTL paper

# Abstract
We propose a novel multi-task learning method that minimizes the effect of negative transfer by allowing asymmetric transfer between the tasks based on task relatedness as well as the amount of individual task losses, which we refer to as Asymmetric Multi-task Learning (AMTL).
To tackle this problem, we couple multiple tasks via a sparse, directed regularization graph, 
that enforces each task parameter to be reconstructed as a sparse combination of other tasks selected based on the task-wise loss. 
We present two dif- ferent algorithms that jointly learn the task pre- dictors as well as the regularization graph. 
The first algorithm solves for the original learning objective using alternative optimization, 
and the second algorithm solves an approximation of it using curriculum learning strategy, that learns one task at a time. We perform experiments on multiple datasets for classification and regression, 
on which we obtain significant improvements in performance over the single task learning and existing multitask learning models.

# Motivation
![motivation](https://photos-5.dropbox.com/t/2/AADq6mfdFk4enlZBo9i9mG0dha0uwny-rwv_tBtJZBd7XA/12/386472799/png/32x32/1/_/1/2/motivation_amtl.png/EPb0-YkDGMwGIAcoBw/C1TgxFYW7jCkRavIhSaJuT9cD_IAk44dPq4VjTv3APQ?dl=0&size=2048x1536&size_mode=3)

# Run code
We have two types of code, regression(run_amtl_regression) and classification(run_amtl_class). I uploaded these codes with example dataset. 

# Details of AMTL
Details of AMTL are described in [AMTL paper][paperlink]
[paperlink]: http://www.jmlr.org/proceedings/papers/v48/leeb16.pdf
