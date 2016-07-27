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

# Run code
We have two types of code, regression(run_amtl_regression) and classification(run_amtl_class). I make these codes with example dataset. 

# Details of AMTL
Details of AMTL are described in [AMTL paper][paperlink]
[paperlink]: http://www.jmlr.org/proceedings/papers/v48/leeb16.pdf
