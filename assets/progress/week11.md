# Week 11

2.1.1. NLP with CNN

CNN 을 Image recognition이 아닌 NLP task (Text Classification task)에 배치했을 때도, LT를 적용할 수 있는지

-	적용할 수 있다!

**2.1.2. LSTM, not CNN**

CNN이 아닌 LSTM에도 LT를 적용할 수 있는지

![plot2](/assets/images/plot2_i133gyzhl.png)

-	IMDB, Epoch = 5로 설정해 간이 training 진행해봄 (학교 GPU 오류 때문)
-	오히려 CNN을 적용했을 때보다 훨씬 압도적으로 lt가 성능이 좋았음
-	Pruning 전보다 더 성능이 좋은 Winning ticket을 찾을 수도 있었음

**2.1.3. One-shot pruning v.s. Iterative Pruning**

반복적으로 특정 정도까지 Pruning한 경우와 한 번에 특정 정도까지 pruning한 경우 중 어느것이 더 성능이 좋은지

```
0.36
86.90286624203821 / 86.92277070063695
87.56966560509554 / 87.54976114649682
87.25119426751591 / 87.26114649681529
87.19148089171973 / 87.609474522293

0.59
86.61425159235668 / 86.06687898089172
87.61942675159236 / 86.92277070063695
86.89291401273886 / 87.54976114649682
86.6640127388535 / 87.62937898089172

0.738
86.5545382165605 / 85.73845541401273
87.1218152866242 / 86.30573248407643
86.90286624203821 / 86.83320063694268
86.6640127388535 / 86.71377388535032

0.832
87.15167197452229 / 85.51950636942675
87.00238853503186 / 85.75835987261146
86.95262738853503 / 86.51472929936305
85.81807324840764 / 87.14171974522293

0.972
77.20939490445859 / 82.65326433121018
75.54737261146497 / 81.5187101910828
77.07006369426752 / 82.52388535031847
76.20421974522293 / 82.45421974522293
```

-	One-shot pruning에서는 Iterative pruning에서와 달리 late rewinding의 성능이 더 좋지 않았다. 하지만, ~80% 수준까지는 Iterative pruning보다 one-shot에서 LT가 random init보다 압도적으로 성능이 좋은 것으로 나타났다.
- Pruning이 90% 후반대로 진입해 거의 모든 parameters가 pruned되면 오히려 random initialization에서 성능이 좋았다.
-	보통 one-shot이 iterative보다 성능이 확연히 떨어진다고 알려져있는데, 초반과 극후반에는 분명 그랬지만 중반에는 오히려 one-shot의 성능이 좋았던 경우도 이 부분에 대해서 더 많이 training시켜보면서 분석해보려 한다. one-shot에서 성능을 잘 낼 수 있으면 iteration을 줄이기 때문에 리소스를 줄일 수 있어서 분석할 가치가 있다.
- 공통적으로, pruning이 많이 진행된 경우에는 수렴 속도가 늦었다. 현실에서 모델 크기를 많이 줄였을 때에는 learning rate를 올려주어야 한다는 suggestion
- 모델이 잘 학습할 수 있도록 Hyperparameters 조정해 학습 재차 진행

**2.1.4. Late rewinding**

초기 weight로 초기화 하는 것보다 일정수준 training이 이루어진 후의 weight을 적용하는 것이 더 성능이 좋은지

-	Iterative pruning에서는 훨씬 좋았지만 one-shot pruning에서는 크게 결정적이지 않았다. Training과정이 적기 때문에 이로 얻을 수 있는 benefit이 적기 때문인 것 같다. 분석해보겠다.

2.1.5. Transfer Learning ``다음주!``

찾은 winning ticket이 같은 task(binary text classification), 다른 dataset(IMDB -> Yelp-2)에 대해 얼만큼의 성능을 내는지, transfer learning이 가능한지
