# Week 9

## IMDb
- 사용한 모델: ``CNN``, ``LSTM``
- global setting
  - train ``28000`` valid ``12000`` test ``10000``
  - initialization: ``Xavier(normal)``
  - criterion: ``BCEWithLogitsLoss``
  - optimizer: ``Adam``
  - learning rate ``1e-3`` batchsize ``64`` epoch ``10``
- [before pruning code](/codes/imdb.ipynb)
- [after random pruning code]()
- [after lt pruning code]()

### Result

#### Before Pruning
| Model | Train Loss (BEST) | Train Acc. (BEST) | Test Loss | Test Acc. |
| ----- | ---------- | ---------- | --------- | --------- |
| CNN | 0.062 | 97.82% | 0.273 | 88.88% |
| LSTM | 0.209 | 91.82% | 0.245 | 90.54% |

#### After Pruning (random init)
| Model | Train Loss (BEST) | Train Acc. (BEST) | Test Loss | Test Acc. |
| ----- | ---------- | ---------- | --------- | --------- |
| CNN | 0.059 | 97.69% | 0.612 | 83.76% |
| LSTM | 0.162 | 93.63% | 0.402 | 84.57% |

#### After Pruning (lt init)
| Model | Train Loss (BEST) | Train Acc. (BEST) | Test Loss | Test Acc. |
| ----- | ---------- | ---------- | --------- | --------- |
| CNN | 0.059 | 97.69% | 0.612 | 83.76% |
| LSTM | 0.162 | 93.63% | 0.402 | 84.57% |

## AG-news
- 사용한 모델: ``CNN``, ``LSTM``
- global setting
  - train ``84000`` valid ``36000`` test ``7600``
  - initialization: ``Xavier(normal)``
  - criterion: ``CrossEntropyLoss``
  - optimizer: ``Adam``
  - learning rate ``1e-3`` batchsize ``64`` epoch ``10``
- [before pruning code](/codes/agnews.ipynb)
- [after random pruning code]()
- [after lt pruning code]()

## Result

#### Before Pruning
| Model | Train Loss (BEST) | Train Acc. (BEST) | Test Loss | Test Acc. |
| ----- | ---------- | ---------- | --------- | --------- |
| CNN | 0.082 | 97.24% | 0.310 | 89.52% |
| LSTM | 0.206 | 92.82% | 0.263 | 91.17% |

#### After Pruning (random init)
| Model | Train Loss (BEST) | Train Acc. (BEST) | Test Loss | Test Acc. |
| ----- | ---------- | ---------- | --------- | --------- |
| CNN | 0.059 | 97.69% | 0.612 | 83.76% |
| LSTM | 0.162 | 93.63% | 0.402 | 84.57% |

#### After Pruning (lt init)
| Model | Train Loss (BEST) | Train Acc. (BEST) | Test Loss | Test Acc. |
| ----- | ---------- | ---------- | --------- | --------- |
| CNN | 0.059 | 97.69% | 0.612 | 83.76% |
| LSTM | 0.162 | 93.63% | 0.402 | 84.57% |
