# TODO: 다음 작업 목록

## 1. 프론트엔드-백엔드 결과 불일치 해결

### 증상
- 백엔드 직접 분석 (WebSocket API) 결과와 프론트엔드(localhost:3000) 화면 결과가 다름
- 예: IMG_5200.MOV 백엔드 Swing 44%/40% vs 프론트엔드에 표시되는 값이 다름

### 점검 항목
- [ ] `lib/websocket.ts` 타입 정의가 백엔드 응답 구조와 일치하는지 확인
  - `GaitMetrics`, `GaitJudgment` 등 인터페이스 검증
  - 특히 `analysis_complete` 메시지의 `results` 필드 매핑
- [ ] `app/test/ai-result/page.tsx` 결과 표시 페이지가 올바른 필드를 참조하는지 확인
- [ ] `store/testSession` Redux 상태에 결과가 올바르게 저장되는지 확인
- [ ] `lib/useAIAnalysis.ts` 훅에서 WebSocket 메시지 파싱 로직 확인
- [ ] 브라우저 캐시/이전 결과가 남아있는지 확인 (hard refresh 테스트)

### 가능한 원인
1. 프론트엔드가 이전 세션의 캐시된 결과를 표시
2. WebSocket 메시지에서 새로 추가된 필드를 파싱하지 않음
3. `lib/websocket.ts`의 수정되지 않은 변경(git status에 modified로 표시)이 충돌
4. 프론트엔드가 judgment 데이터를 다른 경로로 접근

## 2. Step L/R 비대칭 개선 (잔존 이슈)

### 현황
- 일부 영상에서 Step SI 45~52% (정상 <15%)
- 5197: L=0.72m vs R=1.15m
- 5199: L=0.62m vs R=0.97m
- 5204: L=0.41m vs R=0.67m

### 원인 분석
- X-crossing 기반 L/R 할당이 구조적으로 한쪽에 편향될 수 있음
- midpoint 기반 step 거리 계산이 L/R crossing 시점에 따라 비대칭

### 개선 방안 후보
- [ ] Step length 계산을 midpoint 대신 leading foot 좌표 기반으로 변경
- [ ] L/R 교대 강제 (연속 같은 발 이벤트 시 하나 제거)
- [ ] HS 감지 자체를 개선하여 X-crossing 의존도 줄이기 (prominence 추가 튜닝)

## 3. Stride 과대 값 추가 필터링

### 현황
- 5197: L=1.84m, R=2.01m (정상 상한 1.6m 초과)
- 5199: L=1.58m, R=1.93m

### 개선 방안
- [ ] stride 범위 상한을 4.0m → 2.5m로 조정
- [ ] 또는 전체 거리(10m) / stride 수로 계산한 평균과 비교하여 이상치 제거
- [ ] homography 보정 정확도 재검증

## 4. 기타

- [ ] `lib/websocket.ts`의 unstaged 변경 확인 및 필요시 커밋
- [ ] `results/` 폴더에 쌓인 테스트 JSON 파일 정리 (.gitignore 추가 고려)
- [ ] 기각된 솔루션 파일 정리 (`solution_3_smart_foot.py`, `solution_4_optical_flow.py`)
- [ ] `정확도 검증/` 폴더 정리 또는 .gitignore
