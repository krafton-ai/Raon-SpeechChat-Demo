/**
 * protocol.ts — Wire protocol type definitions for full-duplex streaming.
 *
 * Wire format: [1 byte kind] [payload bytes]
 *   0x00 READY     — server → client: session initialized
 *   0x01 AUDIO     — bidirectional: float32 LE PCM samples
 *   0x02 TEXT      — server → client: UTF-8 text delta
 *   0x03 SEQ_TRACE — server → client: full sequence trace
 *   0x04 SEQ_DELTA — server → client: incremental trace delta
 *   0x05 ERROR     — server → client: error message
 *   0x06 CLOSE     — bidirectional: graceful session close
 *   0x07 PING      — bidirectional: keepalive
 *   0x08 PONG      — bidirectional: keepalive response
 */

export enum MessageKind {
  READY = 0x00,
  AUDIO = 0x01,
  TEXT = 0x02,
  SEQ_TRACE = 0x03,
  SEQ_DELTA = 0x04,
  ERROR = 0x05,
  CLOSE = 0x06,
  PING = 0x07,
  PONG = 0x08,
}

export type PromptLanguage = 'eng' | 'kor';
export type TurnMode = 'listen-first' | 'speak-first';
export type SystemPromptStyle =
  | 'generic'
  | 'raon'
  | 'raon_persona'
  | 'raon_persona_context'
  | 'custom';

export interface SessionParams {
  prompt: string;
  prompt_language?: PromptLanguage;
  system_prompt_style?: SystemPromptStyle;
  system_prompt_persona?: string;
  system_prompt_context?: string;
  custom_system_prompt?: string;
  temperature: number;
  top_k: number;
  top_p: number;
  eos_penalty: number;
  bc_penalty: number;
  repetition_penalty: number;
  speaker_mode: string;
}

export interface SessionStats {
  framesSent: number;
  framesRecv: number;
  textChars: number;
}

export interface TimestampedFrame {
  samples: Float32Array;
  timeMs: number; // ms since session start
}

export interface ConversationData {
  transcript: string;
  seqTrace: string;
  inputAudio: Float32Array;
  outputAudio: Float32Array;
  inputFrames: TimestampedFrame[];
  outputFrames: TimestampedFrame[];
  params: Partial<SessionParams>;
  startTime: number;
  endTime: number;
  stats: SessionStats;
}

export type RaonEvent =
  | { type: 'connecting' }
  | { type: 'ready' }
  | { type: 'audio'; data: Float32Array }
  | { type: 'text'; data: string }
  | { type: 'seq_trace'; data: string }
  | { type: 'seq_delta'; data: string }
  | { type: 'error'; message: string }
  | { type: 'close' }
  | { type: 'stats'; data: SessionStats }
  | { type: 'input_rms'; value: number }
  | { type: 'output_rms'; value: number };

export type DuplexEvent = RaonEvent;
