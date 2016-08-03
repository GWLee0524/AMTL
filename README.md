# Asymmetric Multi-Task Learning(AMTL)
+ Giwoong Lee(UNIST), Eunho Yang(KAIST), Sung Ju Hwang(UNIST)

![motivation](https://github.com/BlasterL/image_upload/blob/master/AMTL_ICML2016/amtl_motivation.png)
![mainIdea](https://github.com/BlasterL/image_upload/blob/master/AMTL_ICML2016/amtl_mainidea.png)

## Abstract
We propose a novel multi-task learning method that minimizes the effect of negative transfer by allowing asymmetric transfer between the tasks based on task relatedness as well as the amount of individual task losses, which we refer to as Asymmetric Multi-task Learning (AMTL).
To tackle this problem, we couple multiple tasks via a sparse, directed regularization graph, 
that enforces each task parameter to be reconstructed as a sparse combination of other tasks selected based on the task-wise loss. 
We present two dif- ferent algorithms that jointly learn the task pre- dictors as well as the regularization graph. 
The first algorithm solves for the original learning objective using alternative optimization, 
and the second algorithm solves an approximation of it using curriculum learning strategy, that learns one task at a time. We perform experiments on multiple datasets for classification and regression, 
on which we obtain significant improvements in performance over the single task learning and existing multitask learning models.

## Reference
If you use this code or dataset (such as imbalanced dataset) as part of any published research, please refer the following paper.
```
@inproceedings{lee2016asymmetric,
  title={Asymmetric Multi-task Learning based on Task Relatedness and Confidence},
  author={Lee, Giwoong and Yang, Eunho and others},
  booktitle={Proceedings of The 33rd International Conference on Machine Learning},
  pages={230--238},
  year={2016}
}
```
## Running code
We have two types of code, regression(run_amtl_regression) and classification(run_amtl_class). I uploaded these codes with example dataset. 

## Details of AMTL
Details of AMTL are described in [AMTL paper][paperlink]
[paperlink]: http://www.jmlr.org/proceedings/papers/v48/leeb16.pdf
