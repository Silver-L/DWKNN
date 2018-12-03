-------
DWKNN
-------
#### Distance-weighted k-nearest Neighbor
-------
##### Reference:Gou, J., Du, L., Zhang, Y., & Xiong, T. (2012). A new distance-weighted k-nearest neighbor classifier. J. Inf. Comput. Sci, 9(February), 1429–1436.3
-------
#### Caution:All of the features need to be the same typename!!!

-------
### Task: Detection of bone metastasis in a bone scintigram
#### Result:

| Method        | Average JI    |
| ------------- |:-------------:|
| Graph Cut     | 0.3232        |
| Boosting      | 0.3108        |
| CNN           | 0.2873        |
| ***K-NN***    | ***0.3979***  |
| EM + Map      | 0.1518        |
| SVM           | 0.3598        |
| U-net         | 0.4717        |
