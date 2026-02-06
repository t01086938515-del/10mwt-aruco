/**
 * WebSocket 클라이언트 모듈
 * FastAPI 백엔드와 실시간 통신을 담당
 */

export type WSMessageType =
  | 'calibration'      // ArUco 마커 보정 완료
  | 'frame_data'       // 프레임별 분석 데이터
  | 'crossing_event'   // 라인 통과 이벤트
  | 'analysis_complete' // 분석 완료
  | 'progress'         // 진행률
  | 'error';           // 에러

export interface CalibrationData {
  calibrated: boolean;
  marker_size_m: number;
  marker_distance_m: number;  // 측정 거리
  start_marker_id: number;
  finish_marker_id: number;
  pixels_per_meter: number;
  start_center_px: [number, number] | null;
  finish_center_px: [number, number] | null;
  stored_markers: number[];
  analysis_mode: string;
  start_x: number | null;
  finish_x: number | null;
}

export interface KeypointData {
  x: number;
  y: number;
  conf: number;
}

export interface FrameData {
  frame_number: number;
  frame_idx: number;
  timestamp: number;
  timestamp_s: number;
  keypoints: KeypointData[];
  ankle_left: { x: number; y: number } | null;
  ankle_right: { x: number; y: number } | null;
  center_x: number | null;
  in_zone: boolean;
  frame_image?: string;  // base64 JPEG 이미지
}

export interface CrossingEvent {
  type: 'start' | 'end';
  timestamp: number;
  frame_number: number;
  direction: 'left_to_right' | 'right_to_left';
}

// 판정 결과 타입
export interface JudgmentVariable {
  variable_name: string;
  display_name: string;
  measured_value: number | null;
  unit: string;
  normal_range: string;
  status: '정상' | '상한 초과' | '하한 미달' | 'N/A';
  deviation: number | null;
  direction: '↑' | '↓' | null;
  clinical_comment: string;
  color: 'green' | 'orange' | 'gray';
}

export interface JudgmentPattern {
  pattern: string;
  label: string;
  comment: string;
  color: string;
}

export interface VelocityJudgment {
  speed_mps: number;
  classification: string;
  color: 'green' | 'yellow' | 'orange' | 'red';
  alerts: string[];
}

export interface GaitJudgment {
  variables: JudgmentVariable[];
  patterns: JudgmentPattern[];
  velocity: VelocityJudgment | null;
  reference_note: string;
}

// 백엔드 processor.analysis_results 형식
export interface GaitMetrics {
  // 기본 정보
  elapsed_time_s: number;
  distance_m: number;
  speed_mps: number;
  speed_kmph: number;

  // 보행 지표
  step_count: number | null;
  cadence_spm: number | null;
  step_length_m: number | null;
  stride_length_m: number | null;
  // 좌우 활보장 + SI
  left_stride_length_m: number | null;
  right_stride_length_m: number | null;
  stride_length_si: number | null;

  // 시간 지표
  step_time_s: number | null;
  stride_time_s: number | null;
  left_stride_time_s: number | null;
  right_stride_time_s: number | null;
  stride_time_si: number | null;

  // 좌우 보폭 + SI
  left_step_length_m: number | null;
  right_step_length_m: number | null;
  step_length_si: number | null;

  // 좌우 스텝 시간 + SI
  left_step_time_s: number | null;
  right_step_time_s: number | null;
  step_time_si: number | null;

  // 좌우 스윙 시간 + SI
  left_swing_time_s: number | null;
  right_swing_time_s: number | null;
  swing_time_si: number | null;

  // 좌우 스탠스 시간 + SI
  left_stance_time_s: number | null;
  right_stance_time_s: number | null;
  stance_time_si: number | null;

  // 좌우 스윙/스탠스 비율 + SI
  left_swing_stance_ratio: number | null;
  right_swing_stance_ratio: number | null;
  swing_stance_si: number | null;

  // 스윙/스탠스 백분율 (정상: Swing 40%, Stance 60%)
  left_swing_pct: number | null;
  right_swing_pct: number | null;
  left_stance_pct: number | null;
  right_stance_pct: number | null;

  // 종합 대칭성
  overall_symmetry_index: number | null;

  // 임상 해석
  clinical: {
    fall_risk: string;
    fall_risk_score: number;
    community_ambulation: string;
    speed_category: string;
    recommendations: string[];
  } | null;

  // 판정 결과
  judgment?: GaitJudgment;

  // Step events (enriched with per-step distances)
  step_events?: Array<{
    time: number;
    frame_idx: number;
    leading_foot: 'left' | 'right';
    left_x: number;
    right_x: number;
    peak_y?: number;
    step_num: number;
    step_length_cm: number | null;
    step_time_s: number | null;
    stride_length_cm: number | null;
  }>;

  // Evidence clips
  evidence_clips?: Record<string, { start_s: number; end_s: number; label: string }>;

  // Analysis timeline (per-frame overlay data)
  analysis_timeline?: Array<{
    t: number;      // timestamp_s
    fi: number;     // frame_idx
    lx: number;     // left_ankle_x
    ly: number;     // left_ankle_y
    rx: number;     // right_ankle_x
    ry: number;     // right_ankle_y
    cs: number;     // cumulative_steps
    spd: number;    // instantaneous_speed
    cad: number;    // instantaneous_cadence
    lhx?: number;   // left_heel_x
    lhy?: number;   // left_heel_y
    rhx?: number;   // right_heel_x
    rhy?: number;   // right_heel_y
  }>;

  timeline_meta?: {
    start_t: number;
    finish_t: number;
    fps: number;
    ppm: number;
    total_steps: number;
  };

  analysis_mode: string;
}

// 백엔드 analysis_complete 메시지 형식
export interface AnalysisResult {
  total_frames_analyzed: number;
  results: GaitMetrics | null;
  crossing_events: Array<{
    line: string;
    timestamp_s: number;
    frame_idx: number;
    distance_m?: number;
    elapsed_s?: number;
  }>;
}

export interface WSMessage {
  type: WSMessageType;
  data: CalibrationData | FrameData | CrossingEvent | AnalysisResult | { message: string };
}

type MessageHandler = (message: WSMessage) => void;

export class GaitAnalysisWebSocket {
  private ws: WebSocket | null = null;
  private videoId: string | null = null;
  private handlers: Map<WSMessageType, MessageHandler[]> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 3;
  private baseUrl: string;

  constructor(baseUrl: string = 'ws://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  /**
   * 영상 업로드
   */
  async uploadVideo(file: File): Promise<string> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl.replace('ws', 'http')}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Failed to upload video');
    }

    const data = await response.json();
    this.videoId = data.video_id;
    return data.video_id;
  }

  /**
   * WebSocket 연결
   */
  connect(videoId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.videoId = videoId;
      const wsUrl = `${this.baseUrl}/ws/analyze/${videoId}`;

      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('[WS] Connected to analysis server');
        this.reconnectAttempts = 0;
        resolve();
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WSMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (e) {
          console.error('[WS] Failed to parse message:', e);
        }
      };

      this.ws.onerror = (error) => {
        console.error('[WS] Error:', error);
        reject(error);
      };

      this.ws.onclose = (event) => {
        console.log('[WS] Connection closed:', event.code, event.reason);
        this.handleDisconnect();
      };
    });
  }

  /**
   * 분석 설정 전송
   */
  configure(config: {
    walk_distance?: number;
    marker_size?: number;
    direction?: 'left_to_right' | 'right_to_left' | 'auto';
  }): void {
    this.send({
      type: 'configure',
      ...config,
    });
  }

  /**
   * 분석 시작
   */
  startAnalysis(): void {
    this.send({ type: 'start_analysis' });
  }

  /**
   * 분석 일시정지
   */
  pauseAnalysis(): void {
    this.send({ type: 'pause' });
  }

  /**
   * 분석 리셋
   */
  resetAnalysis(): void {
    this.send({ type: 'reset' });
  }

  /**
   * 이벤트 핸들러 등록
   */
  on(type: WSMessageType, handler: MessageHandler): void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, []);
    }
    this.handlers.get(type)!.push(handler);
  }

  /**
   * 이벤트 핸들러 제거
   */
  off(type: WSMessageType, handler: MessageHandler): void {
    const handlers = this.handlers.get(type);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  /**
   * 연결 종료
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.handlers.clear();
  }

  /**
   * 연결 상태 확인
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private send(data: object): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('[WS] Cannot send - not connected');
    }
  }

  private handleMessage(message: WSMessage): void {
    const handlers = this.handlers.get(message.type);
    if (handlers) {
      handlers.forEach(handler => handler(message));
    }
  }

  private handleDisconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts && this.videoId) {
      this.reconnectAttempts++;
      console.log(`[WS] Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      setTimeout(() => {
        if (this.videoId) {
          this.connect(this.videoId).catch(console.error);
        }
      }, 1000 * this.reconnectAttempts);
    }
  }
}

// 싱글톤 인스턴스
let wsInstance: GaitAnalysisWebSocket | null = null;

export function getWebSocketClient(baseUrl?: string): GaitAnalysisWebSocket {
  if (!wsInstance) {
    wsInstance = new GaitAnalysisWebSocket(baseUrl);
  }
  return wsInstance;
}

export function resetWebSocketClient(): void {
  if (wsInstance) {
    wsInstance.disconnect();
    wsInstance = null;
  }
}
