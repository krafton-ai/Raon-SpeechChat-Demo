/**
 * conversation-export.ts — Export conversation data as a downloadable ZIP.
 *
 * Bundles transcript, metadata, and audio (WAV) into a single .zip file.
 */

import JSZip from 'jszip';
import type { ConversationData, TimestampedFrame } from '../types/protocol';

const SAMPLE_RATE = 24000;

/** Encode Float32 PCM samples as a 16-bit WAV blob. */
function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const dataLength = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataLength, true);
  writeString(view, 8, 'WAVE');

  // fmt subchunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // subchunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true); // byte rate
  view.setUint16(32, numChannels * bytesPerSample, true); // block align
  view.setUint16(34, bitsPerSample, true);

  // data subchunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataLength, true);

  // Convert float32 [-1, 1] to int16 [-32768, 32767]
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

/** Format a timestamp as YYYY-MM-DD-HHmmss. */
function formatTimestamp(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number) => n.toString().padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

/**
 * Build time-aligned stereo channels (left=input, right=output) from timestamped frames.
 *
 * Frames within each channel are concatenated back-to-back (no per-frame jitter).
 * Cross-channel alignment uses the first-frame timestamp of each channel to preserve
 * the relative offset between user speech and assistant response.
 */
function buildStereoChannels(
  inputFrames: TimestampedFrame[],
  outputFrames: TimestampedFrame[],
  sampleRate: number,
): { left: Float32Array; right: Float32Array } {
  if (inputFrames.length === 0 && outputFrames.length === 0) {
    return { left: new Float32Array(0), right: new Float32Array(0) };
  }

  // Determine each channel's start time and total sample count
  const inputStartMs = inputFrames.length > 0 ? inputFrames[0].timeMs : Infinity;
  const outputStartMs = outputFrames.length > 0 ? outputFrames[0].timeMs : Infinity;
  const globalStartMs = Math.min(inputStartMs, outputStartMs);

  const inputOffsetSamples = Math.round(((inputStartMs - globalStartMs) / 1000) * sampleRate);
  const outputOffsetSamples = Math.round(((outputStartMs - globalStartMs) / 1000) * sampleRate);

  const inputTotalSamples = inputFrames.reduce((sum, f) => sum + f.samples.length, 0);
  const outputTotalSamples = outputFrames.reduce((sum, f) => sum + f.samples.length, 0);

  const totalSamples = Math.max(
    inputOffsetSamples + inputTotalSamples,
    outputOffsetSamples + outputTotalSamples,
  );

  const left = new Float32Array(totalSamples);  // input (user mic)
  const right = new Float32Array(totalSamples); // output (assistant)

  // Concatenate back-to-back within each channel
  let pos = inputOffsetSamples;
  for (const frame of inputFrames) {
    left.set(frame.samples, pos);
    pos += frame.samples.length;
  }
  pos = outputOffsetSamples;
  for (const frame of outputFrames) {
    right.set(frame.samples, pos);
    pos += frame.samples.length;
  }

  return { left, right };
}

/** Encode stereo Float32 PCM as a 16-bit WAV blob. */
function encodeStereoWav(left: Float32Array, right: Float32Array, sampleRate: number): Blob {
  const numChannels = 2;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const numSamples = Math.max(left.length, right.length);
  const dataLength = numSamples * numChannels * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataLength, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, bitsPerSample, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataLength, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const l = i < left.length ? Math.max(-1, Math.min(1, left[i])) : 0;
    const r = i < right.length ? Math.max(-1, Math.min(1, right[i])) : 0;
    view.setInt16(offset, l < 0 ? l * 0x8000 : l * 0x7fff, true);
    offset += 2;
    view.setInt16(offset, r < 0 ? r * 0x8000 : r * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/** Export conversation data as a downloadable ZIP file. */
export async function exportConversation(data: ConversationData): Promise<void> {
  const zip = new JSZip();
  const ts = formatTimestamp(data.startTime);

  // Transcript
  if (data.transcript) {
    zip.file('transcript.txt', data.transcript);
  }

  // Sequence trace
  if (data.seqTrace) {
    zip.file('seq_trace.txt', data.seqTrace);
  }

  // Metadata
  const metadata = {
    startTime: new Date(data.startTime).toISOString(),
    endTime: new Date(data.endTime).toISOString(),
    durationSeconds: Math.round((data.endTime - data.startTime) / 1000),
    params: data.params,
    stats: data.stats,
    audio: {
      sampleRate: SAMPLE_RATE,
      inputSamples: data.inputAudio.length,
      outputSamples: data.outputAudio.length,
      inputDurationSeconds: +(data.inputAudio.length / SAMPLE_RATE).toFixed(2),
      outputDurationSeconds: +(data.outputAudio.length / SAMPLE_RATE).toFixed(2),
    },
  };
  zip.file('metadata.json', JSON.stringify(metadata, null, 2));

  // Audio WAV files
  if (data.inputAudio.length > 0) {
    const inputWav = encodeWav(data.inputAudio, SAMPLE_RATE);
    zip.file('input.wav', inputWav);
  }
  if (data.outputAudio.length > 0) {
    const outputWav = encodeWav(data.outputAudio, SAMPLE_RATE);
    zip.file('output.wav', outputWav);
  }

  // Combined stereo WAV (left=user, right=assistant) with timeline alignment
  if (data.inputFrames.length > 0 || data.outputFrames.length > 0) {
    const { left, right } = buildStereoChannels(data.inputFrames, data.outputFrames, SAMPLE_RATE);
    const combinedWav = encodeStereoWav(left, right, SAMPLE_RATE);
    zip.file('combined.wav', combinedWav);
  }

  // Generate and trigger download
  const blob = await zip.generateAsync({ type: 'blob' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `conversation-${ts}.zip`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
