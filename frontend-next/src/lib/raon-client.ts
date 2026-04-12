/**
 * raon-client.ts — WebSocket client and session management for full-duplex streaming.
 *
 * Protocol: 1-byte kind header + payload
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

import { AudioCapture, AudioConstraints, DEFAULT_AUDIO_CONSTRAINTS } from './audio-capture';
import { AudioPlayback } from './audio-playback';
import { MessageKind, SessionParams, RaonEvent, SessionStats, TimestampedFrame } from '../types/protocol';

const LOG = (...args: unknown[]): void => console.log('[fd-demo]', ...args);
const ERR = (...args: unknown[]): void => console.error('[fd-demo]', ...args);

/** Reverse-map from numeric kind to its name, for logging. */
const KIND_NAME: Record<number, string> = Object.fromEntries(
  Object.entries(MessageKind)
    .filter(([, v]) => typeof v === 'number')
    .map(([k, v]) => [v as number, k]),
);

export class RaonClient {
  private _ws: WebSocket | null = null;
  private _capture: AudioCapture;
  private _playback: AudioPlayback;
  private _eventCallback: ((event: RaonEvent) => void) | null = null;
  private _framesSent = 0;
  private _framesRecv = 0;
  private _textChars = 0;
  private _inputFrames: TimestampedFrame[] = [];
  private _outputFrames: TimestampedFrame[] = [];
  private _sessionStartMs = 0;
  private _micThreshold = 0;
  private _audioConstraints: AudioConstraints = DEFAULT_AUDIO_CONSTRAINTS;

  constructor() {
    this._capture = new AudioCapture();
    this._playback = new AudioPlayback();
    LOG('RaonClient created');
  }

  /** Concatenate all buffered input audio into a single Float32Array. */
  get inputAudio(): Float32Array {
    if (this._inputFrames.length === 0) return new Float32Array(0);
    const totalLen = this._inputFrames.reduce((sum, f) => sum + f.samples.length, 0);
    const out = new Float32Array(totalLen);
    let offset = 0;
    for (const frame of this._inputFrames) {
      out.set(frame.samples, offset);
      offset += frame.samples.length;
    }
    return out;
  }

  /** Concatenate all buffered output audio into a single Float32Array. */
  get outputAudio(): Float32Array {
    if (this._outputFrames.length === 0) return new Float32Array(0);
    const totalLen = this._outputFrames.reduce((sum, f) => sum + f.samples.length, 0);
    const out = new Float32Array(totalLen);
    let offset = 0;
    for (const frame of this._outputFrames) {
      out.set(frame.samples, offset);
      offset += frame.samples.length;
    }
    return out;
  }

  /** Timestamped input frames for aligned export. */
  get inputFrames(): TimestampedFrame[] { return this._inputFrames; }

  /** Timestamped output frames for aligned export. */
  get outputFrames(): TimestampedFrame[] { return this._outputFrames; }

  /** True if there is any buffered audio data. */
  get hasAudioData(): boolean {
    return this._inputFrames.length > 0 || this._outputFrames.length > 0;
  }

  /** Register a callback that receives all RaonEvent notifications. */
  onEvent(callback: (event: RaonEvent) => void): void {
    this._eventCallback = callback;
  }

  private _emit(event: RaonEvent): void {
    if (this._eventCallback) this._eventCallback(event);
  }

  private _stats(): SessionStats {
    return {
      framesSent: this._framesSent,
      framesRecv: this._framesRecv,
      textChars: this._textChars,
    };
  }

  connect(url: string, params: Partial<SessionParams> = {}): void {
    const qs = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined || value === null) continue;
      if (typeof value === 'string' && value.trim() === '') continue;
      qs.set(key, String(value));
    }
    const fullUrl = qs.toString() ? `${url}?${qs}` : url;

    LOG('Connecting to', fullUrl);
    this._ws = new WebSocket(fullUrl);
    this._ws.binaryType = 'arraybuffer';
    this._framesSent = 0;
    this._framesRecv = 0;
    this._textChars = 0;
    this._inputFrames = [];
    this._outputFrames = [];
    this._sessionStartMs = performance.now();

    this._ws.onopen = (): void => {
      LOG('WebSocket OPEN');
      this._emit({ type: 'connecting' });
    };

    this._ws.onmessage = async (ev: MessageEvent): Promise<void> => {
      const data = ev.data as unknown;
      if (!(data instanceof ArrayBuffer)) {
        LOG('Received non-ArrayBuffer message:', typeof data, data);
        return;
      }
      if (data.byteLength === 0) {
        LOG('Received empty ArrayBuffer');
        return;
      }

      const kind = new Uint8Array(data, 0, 1)[0];
      const name = KIND_NAME[kind] ?? `UNKNOWN(0x${kind.toString(16)})`;

      if (kind !== MessageKind.AUDIO) {
        LOG(`Received: ${name} (${data.byteLength} bytes)`);
      }

      switch (kind) {
        case MessageKind.READY: {
          LOG('=== READY frame received ===');
          try {
            LOG('Starting playback...');
            this._playback.start();
            this._playback.onRms((rms) => this._emit({ type: 'output_rms', value: rms }));
            LOG('Playback started OK');

            LOG('Requesting mic access...');
            await this._capture.start(this._audioConstraints);
            LOG('Mic capture started OK');
            this._capture.onRms((rms) => this._emit({ type: 'input_rms', value: rms }));
            this._capture.onFrame((frame) => {
              if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;

              // Client-side noise gate: replace below-threshold frames with silence
              let toSend = frame;
              if (this._micThreshold > 0) {
                let sum = 0;
                for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
                const rms = Math.sqrt(sum / frame.length);
                if (rms < this._micThreshold) {
                  toSend = new Float32Array(frame.length); // silence
                }
              }

              this._inputFrames.push({ samples: new Float32Array(toSend), timeMs: performance.now() - this._sessionStartMs });
              const buf = new ArrayBuffer(1 + toSend.length * 4);
              const view = new DataView(buf);
              view.setUint8(0, MessageKind.AUDIO);
              for (let i = 0; i < toSend.length; i++) {
                view.setFloat32(1 + i * 4, toSend[i], /* littleEndian= */ true);
              }
              this._ws.send(buf);
              this._framesSent++;
              if (this._framesSent % 50 === 1) {
                LOG(`Sent frame #${this._framesSent}`);
              }
              this._emit({ type: 'stats', data: this._stats() });
            });
          } catch (err) {
            const e = err as Error;
            ERR('Error in READY handler:', e);
            this._emit({ type: 'error', message: 'Mic access failed: ' + e.message });
          }

          LOG('Emitting ready event');
          this._emit({ type: 'ready' });
          break;
        }

        case MessageKind.AUDIO: {
          // Copy payload to aligned buffer: offset 1 is not 4-byte aligned,
          // which would cause a RangeError when constructing Float32Array directly.
          const audioBytes = new Uint8Array(data, 1);
          const aligned = new ArrayBuffer(audioBytes.length);
          new Uint8Array(aligned).set(audioBytes);
          const floats = new Float32Array(aligned);
          this._outputFrames.push({ samples: new Float32Array(floats), timeMs: performance.now() - this._sessionStartMs });
          this._playback.enqueue(floats);
          this._framesRecv++;
          if (this._framesRecv % 50 === 1) {
            LOG(`Received audio frame #${this._framesRecv} (${floats.length} samples)`);
          }
          this._emit({ type: 'audio', data: floats });
          this._emit({ type: 'stats', data: this._stats() });
          break;
        }

        case MessageKind.TEXT: {
          const text = new TextDecoder().decode(new Uint8Array(data, 1));
          LOG('TEXT:', JSON.stringify(text));
          this._textChars += text.length;
          this._emit({ type: 'text', data: text });
          this._emit({ type: 'stats', data: this._stats() });
          break;
        }

        case MessageKind.SEQ_TRACE: {
          const trace = new TextDecoder().decode(new Uint8Array(data, 1));
          LOG('SEQ_TRACE received (' + trace.length + ' chars)');
          this._emit({ type: 'seq_trace', data: trace });
          break;
        }

        case MessageKind.SEQ_DELTA: {
          const delta = new TextDecoder().decode(new Uint8Array(data, 1));
          this._emit({ type: 'seq_delta', data: delta });
          break;
        }

        case MessageKind.ERROR: {
          const msg = new TextDecoder().decode(new Uint8Array(data, 1));
          ERR('SERVER ERROR:', msg);
          this._emit({ type: 'error', message: msg });
          break;
        }

        case MessageKind.CLOSE: {
          LOG('CLOSE frame received');
          this._cleanup();
          this._emit({ type: 'close' });
          break;
        }

        case MessageKind.PING: {
          LOG('PING received, sending PONG');
          const pong = new Uint8Array([MessageKind.PONG]);
          if (this._ws && this._ws.readyState === WebSocket.OPEN) {
            this._ws.send(pong.buffer);
          }
          break;
        }

        default:
          LOG('Unknown kind:', kind);
          break;
      }
    };

    this._ws.onclose = (e: CloseEvent): void => {
      LOG('WebSocket CLOSED code=' + e.code + ' reason=' + JSON.stringify(e.reason));
      this._cleanup();
      this._emit({ type: 'close' });
    };

    this._ws.onerror = (e: Event): void => {
      ERR('WebSocket ERROR:', e);
      this._emit({ type: 'error', message: 'WebSocket connection error' });
    };
  }

  disconnect(): void {
    LOG('disconnect() called');
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      const closeFrame = new Uint8Array([MessageKind.CLOSE]);
      try { this._ws.send(closeFrame.buffer); } catch (_) {}
    }
    this._cleanup();
    this._emit({ type: 'close' });
    // Clear callback after final emit to prevent late async onclose from re-triggering
    this._eventCallback = null;
  }

  private _cleanup(): void {
    LOG('_cleanup()');
    this._capture.stop();
    this._playback.stop();
    if (this._ws) {
      // Clear handlers before close to prevent async onclose from emitting stale events
      this._ws.onclose = null;
      this._ws.onerror = null;
      this._ws.onmessage = null;
      try { this._ws.close(); } catch (_) {}
      this._ws = null;
    }
  }

  /** Set microphone input gain (1.0 = unity). Applies immediately. */
  setInputGain(gain: number): void {
    this._capture.setGain(gain);
  }

  /** Set client-side noise gate threshold (0 = disabled). */
  setMicThreshold(threshold: number): void {
    this._micThreshold = Math.max(0, threshold);
  }

  /** Set WebRTC audio constraints (applied on next connect). */
  setAudioConstraints(constraints: AudioConstraints): void {
    this._audioConstraints = constraints;
  }

  get framesSent(): number { return this._framesSent; }
  get framesRecv(): number { return this._framesRecv; }
  get textChars(): number { return this._textChars; }
}
