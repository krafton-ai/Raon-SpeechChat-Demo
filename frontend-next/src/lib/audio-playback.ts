/**
 * audio-playback.ts — Gap-free audio playback for full-duplex streaming.
 *
 * Uses frame batching and a pre-buffer gate to absorb network jitter.
 * Incoming 80ms frames are accumulated and scheduled as larger AudioBuffers,
 * reducing node creation overhead and boundary artifacts.
 */

const LOG = (...args: unknown[]): void => console.log('[fd-audio]', ...args);
const ERR = (...args: unknown[]): void => console.error('[fd-audio]', ...args);

const SERVER_SR = 24000;

// ── Batching & pre-buffer ──
const PREBUFFER_FRAMES = 5;               // accumulate 400ms before first playback
const BATCH_MIN_FRAMES = 2;               // batch 160ms during steady playback
const BATCH_FLUSH_MS = 30;                // max hold time before forcing a flush

// ── Scheduling ──
const MAX_SCHEDULE_AHEAD_SEC = 1.25;     // soft cap queueing latency
const EDGE_FADE_SEC = 0.002;             // fade-in on underrun recovery only
const UNDERRUN_GUARD_SEC = 0.020;        // treat <=20ms headroom as underrun
const TARGET_AHEAD_MIN_SEC = 0.30;       // lower startup floor for conversational latency
const TARGET_AHEAD_MAX_SEC = 0.75;
const UNDERRUN_BOOST_STEP_SEC = 0.08;
const UNDERRUN_BOOST_MAX_SEC = 0.25;
const UNDERRUN_BOOST_DECAY_SEC = 0.006;

export class AudioPlayback {
  private _ctx: AudioContext | null = null;
  private _nextPlayTime = 0;
  private _rmsCallback: ((rms: number) => void) | null = null;
  private _enqueueCount = 0;
  private _scheduleCount = 0;
  private _underrunCount = 0;
  private _maxBufferedSec = 0;
  private _lastArrivalAt = 0;
  private _arrivalEwmaSec = 0.08;
  private _arrivalJitterEwmaSec = 0.02;
  private _underrunRecoverySec = 0;
  private _needFadeIn = true;

  // Batching state
  private _pendingFrames: Float32Array[] = [];
  private _prebufferMet = false;
  private _flushTimer: ReturnType<typeof setTimeout> | null = null;

  constructor() {
    LOG('AudioPlayback created');
  }

  onRms(callback: (rms: number) => void): void {
    this._rmsCallback = callback;
  }

  start(): void {
    LOG('AudioPlayback start()');
    try {
      this._ctx = new AudioContext({ sampleRate: SERVER_SR });
      this._ctx.resume().catch((e) => ERR('Playback resume failed:', e));
      this._nextPlayTime = 0;
      this._enqueueCount = 0;
      this._scheduleCount = 0;
      this._underrunCount = 0;
      this._maxBufferedSec = 0;
      this._lastArrivalAt = 0;
      this._arrivalEwmaSec = 0.08;
      this._arrivalJitterEwmaSec = 0.02;
      this._underrunRecoverySec = 0;
      this._needFadeIn = true;
      this._pendingFrames = [];
      this._prebufferMet = false;
      this._clearFlushTimer();
      LOG('Playback AudioContext created, SR:', this._ctx.sampleRate, 'state:', this._ctx.state);
    } catch (err) {
      ERR('Playback AudioContext creation failed:', err);
    }
  }

  /** Push a frame into the batch queue. Playback is scheduled in batches. */
  enqueue(float32Pcm: Float32Array): void {
    if (!this._ctx) return;
    if (float32Pcm.length === 0) return;

    // Report RMS per-frame for responsive meters
    if (this._rmsCallback) {
      let sum = 0;
      for (let i = 0; i < float32Pcm.length; i++) sum += float32Pcm[i] * float32Pcm[i];
      this._rmsCallback(Math.sqrt(sum / float32Pcm.length));
    }

    // Track arrival timing for dynamic target calculation
    const now = this._ctx.currentTime;
    if (this._lastArrivalAt > 0) {
      const delta = Math.max(0, now - this._lastArrivalAt);
      this._arrivalEwmaSec = this._arrivalEwmaSec * 0.9 + delta * 0.1;
      const absDeviation = Math.abs(delta - this._arrivalEwmaSec);
      this._arrivalJitterEwmaSec = this._arrivalJitterEwmaSec * 0.9 + absDeviation * 0.1;
    }
    this._lastArrivalAt = now;

    this._enqueueCount++;
    this._pendingFrames.push(float32Pcm);
    this._maybeFlush();
  }

  private _maybeFlush(): void {
    const needed = this._prebufferMet ? BATCH_MIN_FRAMES : PREBUFFER_FRAMES;

    if (this._pendingFrames.length >= needed) {
      this._flush();
    } else if (!this._flushTimer && this._prebufferMet) {
      // Don't hold frames longer than BATCH_FLUSH_MS during playback
      this._flushTimer = setTimeout(() => {
        this._flushTimer = null;
        if (this._pendingFrames.length > 0) this._flush();
      }, BATCH_FLUSH_MS);
    }
  }

  private _flush(): void {
    this._clearFlushTimer();

    const frames = this._pendingFrames;
    this._pendingFrames = [];
    if (frames.length === 0 || !this._ctx) return;

    // Concatenate all pending frames into one contiguous buffer
    const totalLen = frames.reduce((sum, f) => sum + f.length, 0);
    const combined = new Float32Array(totalLen);
    let offset = 0;
    for (const f of frames) {
      combined.set(f, offset);
      offset += f.length;
    }

    this._prebufferMet = true;
    this._scheduleBuffer(combined);
  }

  private _scheduleBuffer(pcm: Float32Array): void {
    if (!this._ctx) return;

    const buf = this._ctx.createBuffer(1, pcm.length, SERVER_SR);
    buf.getChannelData(0).set(pcm);

    const now = this._ctx.currentTime;

    const dynamicTargetBase = this._arrivalEwmaSec * 2.5 + this._arrivalJitterEwmaSec * 4.0;
    const dynamicTarget = Math.min(
      TARGET_AHEAD_MAX_SEC,
      Math.max(TARGET_AHEAD_MIN_SEC, dynamicTargetBase + this._underrunRecoverySec),
    );

    // Re-prime on first buffer or underrun
    if (this._nextPlayTime <= 0) {
      this._nextPlayTime = now + dynamicTarget;
    } else if (this._nextPlayTime < now + UNDERRUN_GUARD_SEC) {
      this._nextPlayTime = now + dynamicTarget;
      this._underrunCount++;
      this._needFadeIn = true;
      this._underrunRecoverySec = Math.min(
        UNDERRUN_BOOST_MAX_SEC,
        this._underrunRecoverySec + UNDERRUN_BOOST_STEP_SEC,
      );
    } else if (this._nextPlayTime - now > MAX_SCHEDULE_AHEAD_SEC) {
      // Keep cursor monotonic (never rewind) to avoid overlap glitches.
    }

    const startAt = this._nextPlayTime;

    const src = this._ctx.createBufferSource();
    src.buffer = buf;
    const gain = this._ctx.createGain();
    src.connect(gain);
    gain.connect(this._ctx.destination);

    // Only fade-in after underrun recovery to smooth the discontinuity.
    if (this._needFadeIn) {
      const fade = Math.min(EDGE_FADE_SEC, buf.duration * 0.25);
      gain.gain.setValueAtTime(0.0, startAt);
      gain.gain.linearRampToValueAtTime(1.0, startAt + fade);
      this._needFadeIn = false;
    } else {
      gain.gain.setValueAtTime(1.0, startAt);
    }

    src.start(startAt);
    this._nextPlayTime = startAt + buf.duration;
    this._scheduleCount++;
    this._underrunRecoverySec = Math.max(
      0,
      this._underrunRecoverySec - UNDERRUN_BOOST_DECAY_SEC,
    );
    const bufferedAfter = this._nextPlayTime - now;
    if (bufferedAfter > this._maxBufferedSec) {
      this._maxBufferedSec = bufferedAfter;
    }

    if (this._scheduleCount % 20 === 1) {
      LOG(
        `Playback batch #${this._scheduleCount} (${(buf.duration * 1000).toFixed(0)}ms), `
        + `buffered=${bufferedAfter.toFixed(3)}s, target=${dynamicTarget.toFixed(3)}s, `
        + `jitter=${this._arrivalJitterEwmaSec.toFixed(3)}s, `
        + `recovery=${this._underrunRecoverySec.toFixed(3)}s, underruns=${this._underrunCount}`,
      );
    }
  }

  private _clearFlushTimer(): void {
    if (this._flushTimer) {
      clearTimeout(this._flushTimer);
      this._flushTimer = null;
    }
  }

  stop(): void {
    LOG(
      'AudioPlayback stop(), frames enqueued:',
      this._enqueueCount,
      'batches scheduled:',
      this._scheduleCount,
      'underruns:',
      this._underrunCount,
      'max_buffered_s:',
      this._maxBufferedSec.toFixed(3),
    );
    this._clearFlushTimer();
    this._pendingFrames = [];
    this._prebufferMet = false;
    this._nextPlayTime = 0;
    if (this._ctx) {
      this._ctx.close().catch(() => {});
      this._ctx = null;
    }
  }
}
