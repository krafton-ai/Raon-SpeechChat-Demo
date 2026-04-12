'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { ConnectionPanel } from '@/components/ConnectionPanel';
import { AudioMeters } from '@/components/AudioMeters';
import { TranscriptPanel } from '@/components/TranscriptPanel';
import { SessionStatus } from '@/components/SessionStatus';
import { GlassCard } from '@/components/ui/GlassCard';
import type { SessionParams, SessionStats, ConversationData } from '@/types/protocol';
import type { AudioConstraints } from '@/lib/audio-capture';
import { DEFAULT_AUDIO_CONSTRAINTS } from '@/lib/audio-capture';
import { GlassButton } from '@/components/ui/GlassButton';

// RaonClient typed as any until the lib is wired up
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyClient = any;

type StatusKind = 'disconnected' | 'connecting' | 'connected' | 'error';

const EMPTY_STATS: SessionStats = { framesSent: 0, framesRecv: 0, textChars: 0 };

export default function FdDemoPage() {
  const clientRef = useRef<AnyClient>(null);

  const [connected, setConnected]     = useState(false);
  const [status, setStatus]           = useState<StatusKind>('disconnected');
  const [statusText, setStatusText]   = useState('Disconnected');
  const [stats, setStats]             = useState<SessionStats>(EMPTY_STATS);
  const [startTime, setStartTime]     = useState<number | null>(null);

  const [transcript, setTranscript]   = useState('');
  const [seqTrace, setSeqTrace]       = useState('');

  const [inputRms, setInputRms]       = useState(0);
  const [outputRms, setOutputRms]     = useState(0);

  // Mic gain (adjustable while connected)
  const [inputGain, setInputGain] = useState(1.0);

  // Noise gate threshold (adjustable while connected)
  const [micThreshold, setMicThreshold] = useState(0);

  // WebRTC audio constraints (applied at connect time)
  const [audioConstraints, setAudioConstraints] = useState<AudioConstraints>(DEFAULT_AUDIO_CONSTRAINTS);

  // Download state — preserved after disconnect
  const [canDownload, setCanDownload] = useState(false);
  const lastClientRef = useRef<AnyClient>(null);
  const lastParamsRef = useRef<Partial<SessionParams>>({});
  const lastStatsRef = useRef<SessionStats>(EMPTY_STATS);

  // Mobile sidebar toggle
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Apply gain changes to live client
  useEffect(() => {
    if (clientRef.current) clientRef.current.setInputGain(inputGain);
  }, [inputGain]);

  // Apply noise gate threshold to live client
  useEffect(() => {
    if (clientRef.current) clientRef.current.setMicThreshold(micThreshold);
  }, [micThreshold]);

  // Decay meters when disconnected
  useEffect(() => {
    if (connected) return;
    const id = setInterval(() => {
      setInputRms((v) => (v < 0.001 ? 0 : v * 0.8));
      setOutputRms((v) => (v < 0.001 ? 0 : v * 0.8));
    }, 80);
    return () => clearInterval(id);
  }, [connected]);

  const handleDisconnect = useCallback(() => {
    const client = clientRef.current;
    if (client) {
      // Clear ref BEFORE disconnect to prevent re-entrant calls
      // (disconnect() synchronously emits 'close' which re-triggers this handler)
      clientRef.current = null;
      lastClientRef.current = client;
      lastStatsRef.current = { ...stats };
      setCanDownload(client.hasAudioData || transcript.length > 0);
      try { client.disconnect(); } catch { /* ignore */ }
    }
    setConnected(false);
    setStatus('disconnected');
    setStatusText('Disconnected');
    setStartTime(null);
  }, [stats, transcript]);

  const handleConnect = useCallback(
    async (url: string, params: Omit<SessionParams, 'speaker_mode'>) => {
      if (!url) return;

      // Reset display state
      setTranscript('');
      setSeqTrace('');
      setStats(EMPTY_STATS);
      setStatus('connecting');
      setStatusText('Connecting…');

      let RaonClient: new () => AnyClient;
      try {
        const mod = await import('@/lib/raon-client');
        RaonClient = mod.RaonClient;
      } catch {
        setStatus('error');
        setStatusText('Error: raon client module not found');
        return;
      }

      const client = new RaonClient();
      client.setAudioConstraints(audioConstraints);
      client.setMicThreshold(micThreshold);
      client.setInputGain(inputGain);
      clientRef.current = client;

      client.onEvent((event: import('@/types/protocol').RaonEvent) => {
        switch (event.type) {
          case 'connecting':
            setStatus('connecting');
            setStatusText('Waiting for server READY…');
            break;
          case 'ready':
            setConnected(true);
            setStatus('connected');
            setStatusText('Connected · streaming');
            setStartTime(Date.now());
            break;
          case 'text':
            setTranscript((prev) => prev + event.data);
            break;
          case 'seq_trace':
            setSeqTrace(event.data);
            break;
          case 'seq_delta':
            setSeqTrace((prev) => {
              const footer = '='.repeat(120);
              const trimmed = prev.endsWith(footer)
                ? prev.slice(0, -footer.length).replace(/\n$/, '')
                : prev;
              return trimmed.length > 0 ? trimmed + '\n' + event.data : event.data;
            });
            break;
          case 'error':
            setStatus('error');
            setStatusText('Error: ' + event.message);
            setTranscript((prev) => prev + '\n[ERROR] ' + event.message + '\n');
            break;
          case 'input_rms':
            setInputRms(event.value);
            break;
          case 'output_rms':
            setOutputRms(event.value);
            break;
          case 'stats':
            setStats(event.data);
            break;
          case 'close':
            handleDisconnect();
            break;
        }
      });

      const fullParams = { ...params, speaker_mode: 'default' };
      lastParamsRef.current = fullParams;
      setCanDownload(false);
      client.connect(url, fullParams);
    },
    [handleDisconnect, audioConstraints, micThreshold, inputGain],
  );

  const handleDownload = useCallback(async () => {
    const client = lastClientRef.current;
    if (!client) return;
    const { exportConversation } = await import('@/lib/conversation-export');
    const data: ConversationData = {
      transcript,
      seqTrace,
      inputAudio: client.inputAudio,
      outputAudio: client.outputAudio,
      inputFrames: client.inputFrames,
      outputFrames: client.outputFrames,
      params: lastParamsRef.current,
      startTime: startTime ?? Date.now(),
      endTime: Date.now(),
      stats: lastStatsRef.current,
    };
    await exportConversation(data);
  }, [transcript, seqTrace, startTime]);

  return (
    <div
      className="min-h-dvh flex flex-col"
      style={{ fontFamily: 'var(--tahoe-font-sans)' }}
    >
      {/* ── Main content ── */}
      <div
        className="relative z-10 flex flex-col flex-1 w-full max-w-[1280px] mx-auto px-4 py-4 gap-3"
        style={{ minHeight: '100dvh' }}
      >
        {/* Page heading */}
        <header className="text-center pb-1">
          <h1
            className="text-xl font-bold tracking-tight leading-none"
            style={{ color: 'var(--tahoe-text-primary)' }}
          >
            Raon SpeechChat Demo
          </h1>
          <p
            className="text-xs mt-1.5"
            style={{
              color: 'var(--tahoe-text-tertiary)',
              fontFamily: 'var(--tahoe-font-mono)',
            }}
          >
            Real-time streaming conversation · 24 kHz float32 PCM
          </p>
        </header>

        {/* Status bar — full width */}
        <SessionStatus
          connected={connected}
          status={status}
          statusText={statusText}
          stats={stats}
          startTime={startTime}
        />

        {/* Mobile: sidebar toggle button */}
        <div className="flex md:hidden">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="flex items-center gap-2 text-xs font-semibold px-3 py-1.5 rounded-[var(--tahoe-radius-sm)]"
            style={{
              background: 'var(--tahoe-bg-glass)',
              border: '1px solid var(--tahoe-border-glass)',
              color: 'var(--tahoe-text-secondary)',
              backdropFilter: 'blur(24px)',
            }}
            aria-expanded={sidebarOpen}
            aria-controls="sidebar-panel"
          >
            <svg
              className={['w-3.5 h-3.5 transition-transform', sidebarOpen ? 'rotate-180' : ''].join(' ')}
              style={{ transitionDuration: 'var(--tahoe-transition-base)' }}
              viewBox="0 0 16 16"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M1 4h14M1 8h14M1 12h14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" fill="none"/>
            </svg>
            {sidebarOpen ? 'Hide Settings' : 'Show Settings'}
          </button>
        </div>

        {/* Main layout: sidebar + transcript */}
        <div className="flex gap-3 flex-1 min-h-0 flex-col md:flex-row">
          {/* Sidebar */}
          <aside
            id="sidebar-panel"
            className={[
              'w-full md:w-[var(--sidebar-width,280px)] flex-shrink-0 flex flex-col gap-3',
              // On mobile: hide unless open
              sidebarOpen ? 'flex' : 'hidden md:flex',
            ].join(' ')}
          >
            <ConnectionPanel
              onConnect={handleConnect}
              onDisconnect={handleDisconnect}
              connected={connected}
              inputGain={inputGain}
              onGainChange={setInputGain}
              micThreshold={micThreshold}
              onMicThresholdChange={setMicThreshold}
              audioConstraints={audioConstraints}
              onAudioConstraintsChange={setAudioConstraints}
            />

            {/* Quick guide card — desktop only */}
            <GlassCard className="p-4 hidden md:block">
              <p
                className="tahoe-label mb-2.5"
                style={{ color: 'var(--tahoe-text-tertiary)' }}
              >
                Quick Guide
              </p>
              <ul className="space-y-2.5">
                {([
                  ['URL',           'Auto-filled from page origin'],
                  ['System Prompt', 'Controls model behaviour preset'],
                  ['Temperature',   'Higher = more expressive output'],
                  ['EOS Penalty',   'Discourages early stops (0 = off)'],
                ] as [string, string][]).map(([term, desc]) => (
                  <li key={term} className="flex flex-col gap-0.5">
                    <span
                      className="text-[10px] font-semibold"
                      style={{ color: 'var(--tahoe-text-secondary)' }}
                    >
                      {term}
                    </span>
                    <span
                      className="text-[11px] leading-snug"
                      style={{ color: 'var(--tahoe-text-tertiary)' }}
                    >
                      {desc}
                    </span>
                  </li>
                ))}
              </ul>
            </GlassCard>
          </aside>

          {/* Transcript — fills remaining space */}
          <main className="flex-1 flex flex-col min-h-[400px]">
            <TranscriptPanel text={transcript} seqTrace={seqTrace} />
          </main>
        </div>

        {/* Download button — visible after disconnect with data */}
        {canDownload && !connected && (
          <div className="flex justify-center">
            <GlassButton
              variant="accent"
              size="md"
              onClick={handleDownload}
              className="px-6 font-semibold tracking-wide"
            >
              Download Conversation
            </GlassButton>
          </div>
        )}

        {/* Audio meters — full width at bottom */}
        <AudioMeters inputRms={inputRms} outputRms={outputRms} />
      </div>
    </div>
  );
}
