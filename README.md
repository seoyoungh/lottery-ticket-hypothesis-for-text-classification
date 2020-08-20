# Lottery Ticket Hypothesis for Text Classification

This study is being conducted in ``Software Convergence Capstone Design - Spring 2020`` class.

## Overview
ICLR 2019 best paper로 선정된 ​``The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks`` 연구는 network pruning 분야에서 주목을 받았다. 처음 weight initialization에서 사용된 초기 weight를 저장해, 이후 subnetwork의 weight로 다시 넣어주는 간단한 방법이지만 그 성능은 뛰어났다. 하지만, 해당 연구는 ``image classification`` task에 ``CNN``을 적용하는 경우에 대해서만 성능을 검증했다.

다른 연구인 ``PLAYING THE LOTTERY WITH REWARDS AND MULTIPLE LANGUAGES: LOTTERY TICKETS IN RL AND NLP``에서 NLP task에 해당 가설을 적용할 수 있다는 가능성을 보였다. 이 연구에서는 `` language modeling`` task에 ``LSTM``을, ``machine translation`` task에는 ``transformer``를 사용해 두 모델에서 lottery ticket hypothesis를 검증했다.

하지만 아직 NLP 분야에서 주로 사용되는 신경망 모델, 다양한 task에 해당 가설을 범용적으로 적용할 수 있는지는 아직 불분명하다. 따라서, lottery ticket 기법의 NLP 분야에 대한 범용성을 확인하기 위해 연구를 진행하게 되었다.

### Needs
#### Model Compression in NLP
모델 경량화 기법은 CNN 모델을 주로 쓰는 computer vision과 같은 분야에서 연구가 진행되어 왔다. RNN 계열의 모델을 주로 사용하는 NLP 분야에서는 그 필요성이 매우 강조되지는 않았었다. RNN은 다음 입력 데이터 처리를 위해 이전 데이터가 필요하여 병렬화가 어렵다는 한계점 때문에 모델을 거대화하기 어려웠기 때문이다. 하지만, Google이 ``BERT``를 출시한 후 NLP 분야에서도 거대한 모델들이 등장하기 시작했다. 모델의 사전 학습&재학습이 가능해졌고, 뛰어난 성능을 보여주었다. 또한 큰 배치 사이즈가 학습에 효과적이라는 의견이 나오면서 사전 학습에 사용되는 배치 크기가 점점 커지는 추세를 보이고 있다.

메모리 부담을 줄이고, 학습 소요 시간을 줄이고, 모바일/embedded 환경에 배치하기 위해서는 parameters를 줄이는 모델 경량화가 필수적이다. NLP 분야에서도 모델 경량화 연구가 더욱 활발히 진행되어야 한다.


### Goals
선정 모델로  ``text classification``을 진행할 때, pruning 과정에서 ``random initialization``보다 ``lottery ticket initialization`` 방법이 성능이 높음을 검증한다.

추가로, 특정 신경망 모델에서 ``lottery ticket initialization``의 성능을 더 높일 수 있는 방법을 발견하게 되면 이를 제시한다.

## Methods
1) 각 model에 dataset을 학습시킨 후, 모델의 성능을 평가한다.  
2) **Random initialization**으로 pruning을 진행한 후 full network와 성능을 비교한다.  
3) **Winning ticket initialization**으로 pruning을 진행한 후 full network와 성능을 비교한다.
   - Does winning ticket reach the same or higher test accuracy from the full network?

### Models
``CNN``, ``LSTM``

### Task
* **Binary-class text classification**
  - dataset: IMDb Polarity
* **Multi-class text classification**
  - dataset: AG News
* Can **'Late Rewinding'** improve the performance?

### Datasets
| Dataset | Classes | Train samples | Test samples |
|---------|---------|---------------|--------------|
| IMDb Large Movie Review Dataset | 2 | 40,000 | 10,000 |
| AG News | 4 | 120,000 | 7,600 |

[IMDb Dataset source](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
[AG News Datasets source](https://course.fast.ai/datasets)

### Settings
* Training Settings
  - Bias는 0으로 초기화
  - Xavier Initializer 사용
     - LSTM의 weight_hh는 Orthogonal Initializer 사용
  - Adam Optimizer 사용
  - Learning rate: 5e-3 (IMDb), 1e-3 (AG-news)
  - Embedding: Custom (CNN), Glove.6B.100d (LSTM)
  - EPOCH = 10
      - IMDb + CNN은 EPOCH = 20, early stopping, best valid loss가 나온 model state 에서 test accuracy 구함

* Pruning Settings
  - Local Pruning
  - 한 번에 20%씩 20회 pruning 수행
  - weight, bias, embedding 모두 pruning

## Results
[Figures](/results.pdf)

### Without Late Rewinding
|  | Full model performance | Winning ticket performance | Sparsity |
|---------|---------|---------------|--------------|
| **IMDb + CNN** | 87.04 | 87.55 | 36% |
| **AG-news + CNN** | 89.18 | 89.56 | 89.3% |
| **IMDb + LSTM** | 90.60 | 90.24 | 86.6% |
| **AG-news + LSTM** | 91.75 | 91.89 | 48.8% |

### With Late Rewinding
|  | Full model performance | Winning ticket performance | Sparsity |
|---------|---------|---------------|--------------|
| **IMDb + CNN** | 87.04 | 87.64 | 48.8% |
| **AG-news + CNN** | 89.88 | 89.82 | 89.3% |
| **IMDb + LSTM** | 89.96 | 90.49 | 86.6% |
| **AG-news + LSTM** | 91.55 | 92.36 | 73.8% |

## Conclusion
* Lottery Ticket Hypothesis를 CNN은 물론 LSTM에도 적용할 수 있었다.
* Image Recognition task가 아닌 Text Classification task에도 적용이 가능했고, 이를 통해 task에 대한 범용성을 확인할 수 있었다.
* 일반적인 딥러닝 모델은 Over-parameterization화 되어 있을 수 있다.
* 오히려 pruning을 일정 수준 진행해 불필요한 가중치를 없애어 더 좋은 성능 (higher test accuracy)을 내는 model을 찾을 수 있다.
  * 이 때 winning ticket의 sparsity는 dataset / model에 따라 다르다.
* Pruning 후 적절한 sparsity에서 winning ticket을 찾을 수 있다.
* Pruning 과정에서, 초기 initialization에 사용된 weight를 로드해 initialization에 사용하면 더 좋은 성능을 낸다.
* Pruning이 약 90% 이상 진행되면 initialization 방법에 상관 없이 성능이 떨어진다.

## 진행 중인 후속 연구
* The Lottery Hypothesis in Transfer Learning
  - 찾은 winning ticket이 동일한 task, 다른 dataset에 대해 얼만큼의 성능을 내는지

## References
* [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
* [PLAYING THE LOTTERY WITH REWARDS AND MULTIPLE LANGUAGES: LOTTERY TICKETS IN RL AND NLP](https://arxiv.org/abs/1906.02768)
* [Stabilizing the Lottery Ticket Hypothesis](https://arxiv.org/abs/1903.01611v3)

## Researcher
* [Seoyoung Hong](https://github.com/seoyoungh) from Kyunghee Univ.
