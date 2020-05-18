# Week 9

## Finished Works

### training 환경 추가 구축
- 코드 모듈화 완료
- 개발 환경 도커 이미지 구축
- 학교 GPU 서버에 도커 컨테이너 띄움
- 랩탑에서 GPU 서버의 Jupyter로 바로 접속 가능
- 모델 여러개 동시에 돌릴 예정

### 검증
- 두 가지 방법의 pruning을 위한 관련 functions 구축 완료 (드디어 ㅜㅡㅜ)
- IMDB dataset에 대해 두 initialization 방법을 각각 수행한 CNN의 test accuracy 비교
  - workflow
    - training epoch: 10
    - pruning iteration: 20
    - iteration마다 pruning percent: 20%
    - 결국, **전체 network의 20% pruning - pruning 방법에 맞는 initialization - training 10번 - test set 성능 도출** 을 20회 반복하며 pruning iteration마다 performance(test accuracy)를 기록해 비교
  - 20회 진행 후 약 98.8% pruned!

### 결과
- IMDB-CNN result
  - percent of pruned weight
    - [0,0.2,0.36,0.488,0.59,0.672,0.738,0.79,0.832,0.866,0.893,0.914,0.931,0.945,0.956,0.965,0.972,0.977,0.982,0.986,0.988]
  - random initialization
    - [88.2265127388535, 88.16679936305732, 88.40565286624204, 88.02746815286623, 88.05732484076432, 87.75875796178345, 87.86823248407643, 87.83837579617835, 87.35071656050955, 87.03224522292994, 86.5545382165605, 86.36544585987261, 86.46496815286623, 85.95740445859873, 85.85788216560509, 85.1015127388535, 85.49960191082803, 84.35509554140127, 83.92714968152866, 83.25039808917197, 80.1453025477707]
  - lt initialization
    - [89.3312101910828, 88.81369426751591, 88.21656050955414, 87.8781847133758, 87.38057324840764, 87.11186305732484, 86.57444267515923, 86.73367834394905, 86.83320063694268, 86.5545382165605, 86.2062101910828, 85.25079617834395, 85.11146496815286, 84.82285031847134, 83.49920382165605, 82.04617834394905, 82.17555732484077, 81.08081210191082, 80.4140127388535, 79.19984076433121, 74.81090764331209]
  - plot
  - ![plot1](/assets/images/plot1.png)

## To Do
- IMDB-LSTM, AGNEWS-CNN, AGNEWS-LSTM 에 대해 검증
  - 이번 주 안에 무조건 끝낼 것!!!!!!
- BERT로 넘어가기
  - 이번 주에 시작하도록 노력해보기...
- 성능을 높이기 위한 late rewinding 시도해보기
  - 다음 주에 시작하도록 노력해보기...

## Performance Summary
### IMDb
- 사용한 모델: ``CNN``, ``LSTM``
- global setting
  - train ``28000`` valid ``12000`` test ``10000``
  - criterion: ``BCEWithLogitsLoss``
  - optimizer: ``Adam``
  - learning rate ``1e-3`` batchsize ``64`` epoch ``10``
- [before pruning code](/codes/imdb.ipynb)
- [after pruning code - CNN](/codes/imdb-pruning-cnn.ipynb)
- [after pruning code - LSTM]() # not yet

### Result

#### Before Pruning
| Model | Train Loss (BEST) | Train Acc. (BEST) | Test Loss | Test Acc. |
| ----- | ---------- | ---------- | --------- | --------- |
| CNN | 0.062 | 97.82% | 0.273 | 88.88% |
| LSTM | 0.209 | 91.82% | 0.245 | 90.54% |

#### After Pruning (CNN)

### AG-news
- 사용한 모델: ``CNN``, ``LSTM``
- global setting
  - train ``84000`` valid ``36000`` test ``7600``
  - criterion: ``CrossEntropyLoss``
  - optimizer: ``Adam``
  - learning rate ``1e-3`` batchsize ``64`` epoch ``10``
- [before pruning code](/codes/agnews.ipynb)
- [after pruning code - CNN]() # not yet
- [after pruning code - LSTM]() # not yet

### Result

#### Before Pruning
| Model | Train Loss (BEST) | Train Acc. (BEST) | Test Loss | Test Acc. |
| ----- | ---------- | ---------- | --------- | --------- |
| CNN | 0.082 | 97.24% | 0.310 | 89.52% |
| LSTM | 0.206 | 92.82% | 0.263 | 91.17% |

#### After Pruning (CNN)
