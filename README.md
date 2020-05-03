# Lottery Ticket Hypothesis for Text Classification

This study is being conducted in ``Software Convergence Capstone Design - Spring 2020`` class.

## Researcher
* [Seoyoung Hong](https://github.com/seoyoungh) from Kyunghee Univ.

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
2) ``random initialization``으로 pruning을 진행한 후 성능을 비교한다.  
3) ``winning ticket initialization``으로 pruning을 진행한 후 성능을 비교한다.

### Models
``CNN``, ``RNN``, ``LSTM``, ``BERT``

### Task
#### : Text classification (Sentiment Analysis)

##### task 1) Binary-class text classification
* dataset: IMDb Large Movie Review Dataset, ~~Yelp Review Polarity~~

##### task 2) Multi-class text classification
* dataset: AG News, ~~DBPedia~~

### Datasets
| Dataset | Classes | Train samples | Test samples |
|---------|---------|---------------|--------------|
| AG News | 4 | 120,000 | 7,600 |
| IMDb Large Movie Review Dataset | 2 | 250,000 | 250,000 |
| Yelp Review Polarity | 2 | 560,000 | 38,000 |
| ~~DBPedia~~ | 14 | 560,000 | 70,000 |


[IMDb Dataset source](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
[Other Datasets source](https://course.fast.ai/datasets)

## Schedule

### Timeline
* **March**
  - 연구 주제 research ⭕️
  - References 찾기 ⭕️

* **April**
  - 연구 task, model, dataset 선정 ⭕️
  - training - ``CNN``, ``RNN``, ``LSTM``
    - task1 ``IMDb`` ⭕️
    - task2 ``Yelp-5`` 🔜
  - random pruning - ``CNN``, ``RNN``, ``LSTM``
    - task1 ``IMDb`` 🔜
    - task2 ``Yelp-5`` 🔜

* **May**
  - lottery ticket pruning - ``CNN``, ``RNN``, ``LSTM``
    - task1 ``IMDb``
    - task2``Yelp-5``
  - working on ``BERT``
    - training
      - task1
      - task2
    - random pruning
      - task1
      - task2
    - lottery ticket
      - task1
      - task2
  - 각 case별 성능 비교
  - Performace 개선
    - 처음 training할 때 줄 수 있는 better condition 고려
    - hyperparameters 바꾸어보기
    - 더 효과적인 pruning 방법 research (especailly LSTM, BERT)
  - Consider to work with the other two datasets

* **June**
  - ``Conclusion``
      - 최종 setting 채택
      - 최종 성능 도출
  - ``Code Encapsulation/Generalisation``
  - ``Github Deployment``
  - 결과 보고서 작성

### Progress Report

| April |  May  | June  |
|-------|-------|-------|
| | [Week8](/assets/progress/week8.md) | [Week12](/assets/progress/week12.md) |
| | [Week9](/assets/progress/week9.md) | [Week13](/assets/progress/week13.md) |
| [Week6](/assets/progress/week6.md) | [Week10](/assets/progress/week10.md) | [Week14](/assets/progress/week14.md) |
| [Week7 (중간 보고)](/assets/progress/week7.md) | [Week11](/assets/progress/week11.md) | [Week15](/assets/progress/week15.md) |

## References
### Lottery Ticket Hypothesis
* [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) ``The Main Paper``
* [PLAYING THE LOTTERY WITH REWARDS AND MULTIPLE LANGUAGES: LOTTERY TICKETS IN RL AND NLP](https://arxiv.org/abs/1906.02768)
* [Evaluating Lottery Tickets Under Distributional Shifts](https://arxiv.org/abs/1910.12708)
* [Finding Winning Tickets with Limited (or No) Supervision](https://openreview.net/forum?id=SJx_QJHYDB)

### Pruning
* [Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270)
* [Exploring Sparsity in Recurrent Neural Networks](https://arxiv.org/abs/1704.05119)
* [COMPARING REWINDING AND FINE-TUNING IN NEURAL NETWORK PRUNING](https://arxiv.org/abs/2003.02389)
* [C-LSTM: Enabling Efficient LSTM using Structured Compression Techniques on FPGAs](https://arxiv.org/abs/1803.06305)
* [COMPRESSING BERT: STUDYING THE EFFECTS OF WEIGHT PRUNING ON TRANSFER LEARNING](https://arxiv.org/abs/2002.08307)

### Sentiment Analysis
* [Multiclass Sentiment Prediction using Yelp Business Reviews](https://www.semanticscholar.org/paper/Multiclass-Sentiment-Prediction-using-Yelp-Business-Yu/dfa617c7c7e3a53d90c092cef09b2ee1614317a2)

### Posts
* [PyTorch Offical Libary - torchtext](https://pytorch.org/text/index.html)
* [Multi-label Text Classification using BERT](https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d)
* [Multiclass Text Classification using LSTM in Pytorch](https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df)
* [Compressing and regularizing deep neural networks](https://www.oreilly.com/content/compressing-and-regularizing-deep-neural-networks/)
* [딥러닝 모델 압축 방법론과 BERT 압축](https://blog.est.ai/2020/03/딥러닝-모델-압축-방법론과-bert-압축/)
* [torch.nn.utils.prune 모듈로 BERT 다이어트 시키기](https://huffon.github.io/2020/03/15/torch-pruning/)

## License
