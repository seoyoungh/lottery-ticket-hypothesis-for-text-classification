# Week 10

## Finished works
### Issues
- 왜 성능이 잘 나오지 않았을까?
- unpruned 상태에서 validation loss를 살펴보니 overfitting이 의심됨
  - dropout 0.5 부여
  - early stopping 적용, 2번 이상 valid loss가 나오면 epoch iteration stop
  - test performance는 best valid loss를 낸 모델 state에서 평가
- learning rate
  - learning rate 1e-3 -> 5e-3 조정
  - unpruned에서는 비교적 lr을 작게 설정하는게, pruned에서는 비교적 크게 설정하는게 성능이 좋았음
  - 중간 크기인 5e-3 채택
  - pruning 후에는 수렴이 확실히 늦다. -> 다른 논문에서도 제기된 문제점
- EPOCH는 충분히 학습되도록 20으로 설정
- embedding은 초기화하지 않았었음
  - embedding weight는 두 케이스 동일하게 처음 weight으로 부여

### Performance
- [Performance](/assets/codes/CNN_Results.ipynb)

### CNN과, LSTM에 집중해 다양한 condition 주고 성능 검증하기
- Why? BERT는 다른 방법으로 보통 pruning (모델 자체가 아예 pre-trained model - 초기 weight 애매함)
- CNN, LSTM에 집중해서 다양한 condition 줘보기.
  - 1. real initial state v.s. late resetting
    - late resetting after the first epoch state v.s. after the last epoch state
  - 2. local pruning v.s. global pruning
    - 이번에 진행한건 모든 레이어에 대해 일괄적으로 20% pruning
    - 하지만 parameters number 차이도 있어서 global pruning 해보기
    - L1 norm unstructured pruning 시도 -> 성능 높아지는지?
  - 3. one-shot pruning v.s. iterative pruning
    - 한번에 50% pruning하냐, 한번에 20%씩 학습을 반복하며 최종 50% pruning 하냐
    - 보편적으로 당연히 iterative pruning이 더 성능이 좋지만, computing power를 고려하면 one-shot pruning이 훨씬 효율적
    - iterative pruning 만큼은 아니지만, 얼마만큼의 성능을 낼 수 있는지 검증
    - one-shot pruning에서도 lottery ticket이 더 성능이 좋은지 검증
    - iterative 보다 one-shot에서 눈에 띄게 더 성능이 좋다거나 (바람..?)
- transfer learning 시도해보기
  - 같은 binary classification task일 때, pruned model에 대해 dataset간의 transfer learning 적용 가능한지

### Default setting
   1. optimizer: Adam / lr: 5e-3 / dropout 0.5
   2. initialization
      - CNN: Xavier Normal
      - LSTM: Xavier Normal + Orthogonal
   3. smallest-magnitude pruning (sparse pruning)
   4. local pruning: 모든 layer에 동일하게 20%씩 pruning

### late resetting (late rewinding)
- late resetting after the first epoch state v.s. after the last epoch state
- training을 fully하고 그 때의 weight를 저장해 초기화 시켜주면 성능이 더욱 좋았음

## Future Tasks
### 5월
* LSTM pruning 진행
* [local pruning v.s. global pruning](https://tutorials.pytorch.kr/intermediate/pruning_tutorial.html)

### 6월
* one-shot pruning v.s. iterative pruning
* 같은 task에 대해 transfer learning이 가능한지
- IMDB -> YELP-2
- random pruning, lt pruning, transferred winning ticket 적용 비교
  - sparsity: 100 41.1 16.9 7.0 2.9 1.2 0.5 0.2
