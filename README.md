# KUDataRepresentation

- 원본 시계열 데이터를 입력으로 받아 각 관측치에 대한 representation vector를 도출하는 time series representation에 대한 설명
- 입력 데이터 형태 : (num_of_instance x input_dims x seq_len) 차원의 다변량 시계열 데이터(multivariate time-series data)
<br>

**Time series representation 사용 시, 설정해야하는 값**
* **model** : ['ts2vec', 'ts_tcc', 'rae_mepc', 'stoc'] 중 선택
* **training** : 모델 학습 여부, [True, False] 중 선택, 학습 완료된 모델이 저장되어 있다면 False 선택
* **best_model_path** : 학습 완료된 모델을 저장할 경로

* **시계열 representation 모델 hyperparameter :** 아래에 자세히 설명.
  * TS2Vec hyperparameter 
  * TS-TCC hyperparameter 
  * RAE-MEPC hyperparameter
  * STOC hyperparameter
<br>

#### 시계열 representation 모델 hyperparameter <br>

#### 1. TS2Vec
- **input_dim** : 데이터의 변수 개수, int
- **repr_dim** : data representation 차원, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
- **hidden_dim** : encoder의 hidden dimension, int(default: 64, 범위: 1 이상, default 값 사용 권장)
- **num_epochs** : 학습 epoch 횟수, int(default: 50, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 512, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
<br>

#### 2. TS-TCC
- **input_dim** : 데이터의 변수 개수, int
- **repr_dim** : data representation 차원, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
- **hidden_dim** : temporal / contextual contrasting 모듈의 hidden dimension, int(default: 100, 범위: 1 이상, default 값 사용 권장)
- **timesteps** : temporal contrasting 모듈에서 미래 예측할 시점의 길이, int(default: 6, 범위: 1 이상)
- **num_epochs** : 학습 epoch 횟수, int(default: 50, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 512, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
- **jitter_scale_ratio** : time series data augementation 중 weak augementation의 강도, float(default: 1.1, default 값 사용 권장)
- **jitter_ratio** : time series data augementation 중 strong augementation의 강도, float(default: 0.8, default 값 사용 권장)
- **max_seg** : strong augementation에서 permutation 진행시 데이터의 최대 분할 개수, int(default: 8, default 값 사용 권장)
<br>

#### 3.	RAE-MEPC
- **window_size** : 모델의 input sequence 길이, int(default: 32, 범위: 0 이상 & 원래 데이터의 sequence 길이 이하)
- **input_dim** : 데이터의 변수 개수, int
- **repr_dim** : data representation 차원, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
- **enc_nlayers** : multi-resolution encoder를 구성하는 sub-encoder의 개수, int(default: 3, 범위: 1 이상, default 값 사용 권장)
- **dec_nlayers** : multi-resolution decoder를 구성하는 sub-decoder의 개수, int(default: 3, 범위: 1 이상, default 값 사용 권장)
- **tau** : multi-resolution encoder 및 decoder의 resolution를 조절하는 값, int(default: 4, 범위: 2 이상, default 값 사용 권장)
- **num_epochs** : 학습 epoch 횟수, int(default: 50, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 512, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
<br>

#### 4.	STOC
- **window_size** : 모델의 input sequence 길이, int(default: 32, 범위: 0 이상 & 원래 데이터의 sequence 길이 이하)
- **input_dim** : 데이터의 변수 개수, int
- **repr_dim** : data representation 차원, int(default: 64, 범위: 1 이상, 2의 지수로 설정 권장)
- **hidden_dim** : encoder의 hidden dimension, int(default: 256, 범위: 1 이상, default 값 사용 권장)
- **forecast_step** : 미래 시계열 데이터에 대하여 예측할 시점의 길이, int(default: 6, 범위: 1 이상)
- **num_epochs** : 학습 epoch 횟수, int(default: 50, 범위: 1 이상)
- **batch_size** : batch 크기, int(default: 512, 범위: 1 이상, 컴퓨터 사양에 적합하게 설정)
- **lr** : learning rate, float(default: 0.001, 범위: 0.1 이하)
- **device** : 학습 환경, (default: 'cuda', ['cuda', 'cpu'] 중 선택)
- **patience** : 예측 모델 학습 시, 사전 설정한 epoch 동안 loss가 감소하지 않으면 학습 조기 중단, int(default: 10, 범위: 1 이상 num_epochs 미만)
<br>
