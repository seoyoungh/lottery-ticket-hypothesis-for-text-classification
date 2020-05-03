# Week 7 (중간 보고)

## finished works
- [x] task1 - IMDb training
- [x] task1 - Yelp-polarity training
- [x] task2 - AG-news training

- IMDb case의 test accuracy가 생각보다 낮게 나와, task1에 대해 Yelp-polarity dataset을 추가로 학습시켜 보았습니다.
- 우선 IMDb accuracy가 많이 낮은 것은 아니여서, 이 데이터셋으로 진행하려 합니다.
- 하지만, IMDb dataset 자체가 다른 데이터셋에 비해 작아 리소스가 남는다면 Yelp-polarity에 대한 검증도 추후에 이루어지면 좋을 것 같습니다.
- task2를 진행할 때, class가 증가하면서 고쳐야할 부분이 많아 계속 오류가 발생했습니다. 이 오류를 해결하다가 진행이 조금 늦어졌습니다.
- 가장 오래 걸리는 작업이 importing data, model implementation인데 생각보다 일찍 끝났습니다. pruning 자체는 어렵지 않아 앞으로는 속도 내서 진행하려 합니다.
- vanilla RNN의 accuracy가 상당히 낮은데, (특히 multi-class에서) 이는 모델 구조 자체의 Performace가 떨어지기 때문입니다. 사실 단독으로 RNN을 쓰는 경우는 최근 들어 거의 없습니다. RNN이라 하면 통상 기존 RNN의 Vanishing gradients a문제를 개선한 LSTM을 의미합니다. 따라서 연구에서 제외시키려 합니다. **CNN, LSTM, BERT에 대해 진행하겠습니다.**
- local pruning이 아닌 global pruning으로 진행하려 합니다.
- 모델 원문에서는 iterative pruning이 더 성능이 좋았습니다. one-shot pruning, iterative pruning 모두 시도해보려 합니다.

## binary-class text classification
[base code source](https://github.com/prakashpandey9/Text-Classification-Pytorch)

### IMDb
  - 사용한 모델: ``CNN``, ~~RNN~~, ``LSTM``
  - [before pruning code](/codes/IMDb.ipynb)
  - [after random pruning code](/codes/IMDb_random.ipynb)
  - global setting
    - train 17500 validation 7500 test 25000
    - learning rate 2e-5 batchsize 32 epoch 10
  - result

#### Before Pruning
pruning
| Model | Train Loss (BEST) | Train Acc. (BEST) | Test Loss | Test Acc. |
| ----- | ---------- | ---------- | --------- | --------- |
| CNN | 0.116 | 95.77% | 0.471 | 84.07% |
| ~~RNN~~ | 0.640 | 64.31% | 0.754 | 58.12% |
| LSTM | 0.117 | 95.74% | 0.545 | 84.06% |

#### After Random Pruning ``진행중``
- sparsity: 20%
- sparsity: 36%에 대해서도 수행해봤는데, overfitting 문제가 있는 것 같아 더이상 pruning하지 않았습니다. 수행시간에도 큰 차이가 없었는데, 이 방식으로 여러번 수행하면 안 되는 것 같습니다. iterative pruning 방법을 찾아볼 계획입니다.

| Model | Train Loss (BEST) | Train Acc. (BEST) | Test Loss | Test Acc. |
| ----- | ---------- | ---------- | --------- | --------- |
| CNN | 0.059 | 97.69% | 0.612 | 83.76% |
| LSTM | 0.162 | 93.63% | 0.402 | 84.57% |

### Yelp-polarity
  - 사용한 모델: ``CNN``, ~~RNN~~, ``LSTM``
  - [before pruning code](/codes/Yelp-polarity.ipynb)
  - global setting
    - train 392000 validation 168000 test 38000
    - learning rate 2e-5 batchsize 32 epoch 5
  - result

| Model | Train Loss (BEST) | Train Acc. (BEST) | Test Loss | Test Acc. |
| ----- | ---------- | ---------- | --------- | --------- |
| CNN | 0.219 | 91.84% | 0.192 | 92.78% |
| ~~RNN~~ | 0.515 | 74.78% | 0.458 | 77.88% |
| LSTM | 0.075 | 97.21% | 0.145 | 95.01% |

## multi-class text classification
[base code source](https://github.com/bentrevett/pytorch-sentiment-analysis)

### AG-news
  - 사용한 모델: ``CNN``, ~~RNN~~, ``LSTM``
  - [before pruning code](/codes/AG-news.ipynb)
  - global setting
    - train 84000 validation 36000 test 7600
      - learning rate ``Adam`` batchsize 64 epoch 5
 - result

| Model | Train Loss (BEST) | Train Acc. (BEST) | Test Loss | Test Acc. |
| ----- | ---------- | ---------- | --------- | --------- |
| CNN | 0.069 | 97.65% | 0.346 | 90.45% |
| ~~RNN~~ | 1.385 | 25.31% | 1.386 | 25.30% |
| LSTM | 0.234 | 91.88% | 0.250 | 91.20% |

## To Do
- [ ] IMDb
  - [x] training
  - [x] random pruning (one-shot)
    - 진행중 [code](/codes/IMDb_random.ipynb))
  - [ ] lt pruning
- [ ] Yelp-polarity
  - [x] training
- [ ] AG-news
  - [x] training
  - [ ] random pruning
  - [ ] lt pruning
- [ ] Bert (task1 - IMDb)
  - [ ] training
  - [ ] random pruning
  - [ ] lt pruning
- [ ] Bert (task2 - AG-news)
  - [ ] training
  - [ ] random pruning
  - [ ] lt pruning
- [ ] try iterative pruning & late rewinding for all cases
