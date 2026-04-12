/**
 * audio-capture.ts — Microphone capture for full-duplex streaming.
 *
 * Captures mono audio from the user's microphone, resamples to 24kHz,
 * and dispatches fixed-size frames (1920 samples = 80ms) via onFrame().
 *
 * Note: ScriptProcessorNode is deprecated. A future migration to
 * AudioWorklet (AudioWorkletProcessor) would improve performance and
 * eliminate the main-thread processing overhead.
 */

const LOG = (...args: unknown[]): void => console.log('[fd-audio]', ...args);
const ERR = (...args: unknown[]): void => console.error('[fd-audio]', ...args);

export const SERVER_SR = 24000;
export const FRAME_SIZE = 1920; // 80ms at 24kHz — matches model's native frame size
const CAPTURE_BUFFER = 4096;

/**
 * Linearly interpolates inputBuffer (at inputSampleRate) down to SERVER_SR (24kHz).
 * Returns the original buffer unchanged if no resampling is needed.
 */
export function resampleTo24k(
  inputBuffer: Float32Array,
  inputSampleRate: number,
): Float32Array {
  if (inputSampleRate === SERVER_SR) return inputBuffer;
  const ratio = inputSampleRate / SERVER_SR;
  const outLen = Math.round(inputBuffer.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * ratio;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(lo + 1, inputBuffer.length - 1);
    const frac = srcIdx - lo;
    out[i] = inputBuffer[lo] * (1 - frac) + inputBuffer[hi] * frac;
  }
  return out;
}

export interface AudioConstraints {
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
}

export const DEFAULT_AUDIO_CONSTRAINTS: AudioConstraints = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

export class AudioCapture {
  private _callback: ((frame: Float32Array) => void) | null = null;
  private _rmsCallback: ((rms: number) => void) | null = null;
  private _ctx: AudioContext | null = null;
  private _stream: MediaStream | null = null;
  // ScriptProcessorNode is deprecated but widely supported. Migrate to
  // AudioWorkletNode when browser support requirements allow.
  private _scriptNode: ScriptProcessorNode | null = null;
  private _micSrc: MediaStreamAudioSourceNode | null = null;
  private _accumulator: Float32Array = new Float32Array(0);
  private _frameCount = 0;
  private _gain = 1.0;

  constructor() {
    LOG('AudioCapture created');
  }

  /** Set microphone input gain (1.0 = unity, 2.0 = +6dB, 0.5 = -6dB). */
  setGain(gain: number): void {
    this._gain = Math.max(0, gain);
  }

  onFrame(callback: (frame: Float32Array) => void): void {
    this._callback = callback;
  }

  onRms(callback: (rms: number) => void): void {
    this._rmsCallback = callback;
  }

  async start(constraints: AudioConstraints = DEFAULT_AUDIO_CONSTRAINTS): Promise<void> {
    LOG('Requesting getUserMedia...', constraints);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const msg = 'Mic access requires HTTPS. Use cloudflared tunnel or access via https://.';
      ERR(msg);
      throw new Error(msg);
    }

    try {
      this._stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: constraints.echoCancellation,
          noiseSuppression: constraints.noiseSuppression,
          autoGainControl: constraints.autoGainControl,
        },
      });
      LOG('getUserMedia OK, tracks:', this._stream.getAudioTracks().map((t) => t.label));
    } catch (err) {
      const e = err as Error;
      ERR('getUserMedia FAILED:', e.name, e.message);
      throw err;
    }

    try {
      this._ctx = new AudioContext();
      LOG('AudioContext created, sampleRate:', this._ctx.sampleRate, 'state:', this._ctx.state);
      await this._ctx.resume();
      LOG('AudioContext resumed, state:', this._ctx.state);
    } catch (err) {
      ERR('AudioContext creation failed:', err);
      throw err;
    }

    const micSR = this._ctx.sampleRate;
    LOG('Mic sample rate:', micSR, '→ resampling to', SERVER_SR);

    this._micSrc = this._ctx.createMediaStreamSource(this._stream);
    // createScriptProcessor is deprecated; see note at top of file.
    this._scriptNode = this._ctx.createScriptProcessor(CAPTURE_BUFFER, 1, 1);

    this._scriptNode.onaudioprocess = (e: AudioProcessingEvent): void => {
      let pcm: Float32Array = new Float32Array(e.inputBuffer.getChannelData(0));

      if (micSR !== SERVER_SR) {
        pcm = resampleTo24k(pcm, micSR);
      }

      // Apply input gain
      if (this._gain !== 1.0) {
        for (let i = 0; i < pcm.length; i++) pcm[i] *= this._gain;
      }

      if (this._rmsCallback) {
        let sum = 0;
        for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
        this._rmsCallback(Math.sqrt(sum / pcm.length));
      }

      const prev = this._accumulator;
      const combined = new Float32Array(prev.length + pcm.length);
      combined.set(prev);
      combined.set(pcm, prev.length);
      this._accumulator = combined;

      while (this._accumulator.length >= FRAME_SIZE) {
        const frame = this._accumulator.slice(0, FRAME_SIZE);
        this._accumulator = this._accumulator.slice(FRAME_SIZE);
        this._frameCount++;
        if (this._callback) this._callback(frame);
      }
    };

    this._micSrc.connect(this._scriptNode);
    this._scriptNode.connect(this._ctx.destination);
    LOG('Capture pipeline connected: mic → scriptNode → destination');
    LOG('AudioCapture start() complete');
  }

  stop(): void {
    LOG('AudioCapture stop(), frames dispatched:', this._frameCount);
    if (this._scriptNode) {
      try { this._scriptNode.disconnect(); } catch (_) {}
      this._scriptNode = null;
    }
    if (this._micSrc) {
      try { this._micSrc.disconnect(); } catch (_) {}
      this._micSrc = null;
    }
    if (this._stream) {
      this._stream.getTracks().forEach((t) => t.stop());
      this._stream = null;
    }
    if (this._ctx) {
      this._ctx.close().catch(() => {});
      this._ctx = null;
    }
    this._accumulator = new Float32Array(0);
    this._frameCount = 0;
  }
}
