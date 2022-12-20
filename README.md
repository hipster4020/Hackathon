# Hackathon
2022 한국어 멀티세션 대화 데이터 해커톤 활용<br><br>

# 👉🏻 model train 환경
Google Colab Pro Plus<br>
gpu info : A100-SXM4-40GB<br><br>

# 👉🏻 실행 가능한 형태의 훈련 코드, 인퍼 코드
## run command
<b>src/training/main.ipynb</b><br>
◉ <b>Install</b> 설치<br>
◉ <b>Predict</b> 결과 추출<br><br>

# 👉🏻 최종 훈련된 모델
<b>model/</b><br><br>

# 👉🏻 훈련 로그
https://wandb.ai/psh_pat/hackathon/runs/45xlkwzn?workspace=user-psh_pat<br><br>

# 👉🏻 훈련 모델 구조에 대한 설명 자료
KoGPT2 (한국어 GPT-2) Ver 2.0 finetuning<br>
https://github.com/SKT-AI/KoGPT2<br><br>

# 👉🏻 멀티세션 대화 데이터 피드백 내용
대화로 이어지는 데이터의 질이 좋고 데이터의 양이 많다.<br>
다만, 페르소나 정보, 이전 대화와의 지난 시간 정보, 요약 정보 등 고려되어야 할 피쳐가 너무 많아 큰 규모의 pretrained model이 필요할 듯 한데, 일반 개발자들은 그런 규모를 감당할 수 있는 서버를 보유 할 수 없으니 굉장히 힘들었다.<br>
좀 더 큰 모델인 T5나 kakao/kogpt 모델을 사용하려고 하였으나 colab pro plus 환경에서도 메모리가 감당되지 않아 skt/koGPT2를 사용하였는데 성능이 굉장히 좋지 않게 나온 듯 하다.