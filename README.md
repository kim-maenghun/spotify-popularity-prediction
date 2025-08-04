# 🎧 Spotify Popularity Prediction

> 캐글 [Spotify - Hit Predictor](https://www.kaggle.com/competitions/spotify-da-ml) 대회 참가 프로젝트입니다.  
> Spotify 음원 데이터를 바탕으로 트랙의 `popularity` 점수를 예측하는 회귀 문제를 다룹니다.

## 🧠 주요 모델

- XGBoost Regressor (tree_method='hist', grow_policy='lossguide')
- RandomizedSearchCV로 하이퍼파라미터 튜닝
- OOF Cross-validation (KFold, n=5)

> 🥇 **최종 순위: 4등 / 전체 참가자 수**  
> 🧠 XGBoost + Feature Engineering 기반 모델  
> 📉 RMSE: 0.2421 (Cross-validation 기준)


## 🧪 주요 전처리 및 모델링 전략

### 🔹 전처리
- `track_genre`: Target Encoding (with smoothing)
- log 변환 (`log_duration`)
- 파생 변수 생성 (`dance_energy`, `loudness_tempo`)
- Polynomial 상호작용 변수 (상위 15개 선택)
- 이상치 클리핑 (1%, 99% quantile 기준)
- StandardScaler로 스케일링

### 🔹 모델 튜닝
```python
param_dist = {
    'n_estimators': [600,800,1000],
    'learning_rate': [0.01,0.02,0.03],
    ...
}
