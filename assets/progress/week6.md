# Week 6

## Finished Works

1. 환경 구축
    - TPU 사용해봄! 훨씬 빠름!
    - torch.device module에서는 아직 parameter로 tpu 줄 수 없음 :<
    - 해당 코드 넣어야 하는 경우에는 GPU 사용함

2. DATASET 선정 & 수집 완료
    - TASK 두개에 대해 각각 2개의 DATASET 선정
        - 우선 각각의 task에 대해 데이터셋 한개씩만 사용
          - binary-class text classification: IMDb
          - multi-class text classification: Yelp-5
        - 시간적 여유가 있다면 cycle이 끝난 후 나머지 dataset에 대해서도 연구 진행

3. Model 구현 완료
    - Opensource 기반으로 model implementation 완료
      - References
        - [Text-Classification-Pytorch](https://github.com/prakashpandey9/Text-Classification-Pytorch)
        - [pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis)
    - training할 때 함수로 불러오면 됨
    - 정확도가 높게 나오지 않는다면 이후에 hyperparameter 및 구조 수정해나갈 예정

4. 우선 IMDb Dataset에 대해 training 진행
    - 사용한 모델: ``RNN``, ``CNN``, ``LSTM``
    - [code](/codes/IMDb-new-version.ipynb)
    - global setting
      - train 17500 validation 7500 test 25000
      - learning rate 2e-5 batchsize 32 epoch 10
   - result

| Model | Train Loss (BEST) | Train Acc. (BEST) | Test Loss | Test Acc. |
| ----- | ---------- | ---------- | --------- | --------- |
| CNN | 0.116 | 95.77% | 0.471 | 84.07% |
| RNN | 0.640 | 64.31% | 0.754 | 58.12% |
| LSTM | 0.117 | 95.74% | 0.545 | 84.06% |

5. BERT Pruning
    - naive하게 프루닝 방법 살펴봄
        - pytorch에서 제공하는 prune 기능 사용
        - 30% sparsity
    - Pre-trained model이라 모델 fine-tuning 필요
    - but, fine-tuning 없이 단순히 사전학습된 모델에 대해서만 pruning 진행해봄 (정확도 비교 X)
    - [code](/codes/testing-bert-pruning.ipynb)

6. [Timeline 수정](https://github.com/seoyoungh/lottery-ticket-hypothesis-for-text-classification)

## Issues

1. Data preprocessing issue
    - Main subject는 ‘task 해결’이 아닌 ‘pruning’, 성능을 극도로 높일 필요는 없음
    - vocabulary 구축하면서 자체가 해결해주는 정도로 일단 넘어감
        - 많이 쓰이는 단어 위주로 분석, vocab에 없는 단어는 ``<sth>``으로 처리하는 방식  

2. dataset source issue
    - 이번 연구에 사용될 모든 데이터셋은 csv 형태의, text/label만 있는 데이터로 수집 완료
    - but, 사용하려는 모든 dataset에 Pytorch에서 torch.datasets 라이브러리 제공
    - data를 불러오는 소요시간은 짧음, libaray에서 제공하는 method 사용도 훨씬 편함
    - but, colab의 runtime은 최대 12시간, cloud하게 계속 불러오는게 효율적인지?
    - 그리고 ``yelp-5``는 계속 존재하지 않는 데이터셋이라고 나옴
    - 수집한 dataset을 직접적으로 import 하는 방향으로 수정해나갈 예정

3. Encapsulation의 필요성
    - 장기적 관점에서 내 리소스 줄이기
    - Opensource로 배포하기 위해선 package화 해야함
    - main.py
      - loadData.py
        - /datasets
      - loadModel.py
        - /models
        - 모델에 따라 필요로하는 parameters 다름에 유의
        - BERT의 경우 사전 학습 필요

4. BERT의 학습시간 issue
    - BERT의 학습시간 대비 colab의 runtime이 짧음
        - BERT 학습을 위해 학교 GPU 대여 신청 완료, 5/6 부터 사용 가능

## To do
- RNN 성능 높일 방법 찾아보기
- Yelp-5 Dataset 학습 마치기 (CNN, RNN, LSTM)
- trained models에 대해 random pruning 진행
