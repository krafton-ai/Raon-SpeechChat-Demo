'use client';

import { useState, useEffect } from 'react';
import type { SessionStats } from '@/types/protocol';
import { GlassPanel } from '@/components/ui/GlassPanel';
import { ThemeToggle } from '@/components/ui/ThemeToggle';

type StatusKind = 'disconnected' | 'connecting' | 'connected' | 'error';

interface SessionStatusProps {
  connected: boolean;
  status: StatusKind;
  statusText: string;
  stats: SessionStats;
  startTime: number | null;
}

function useDuration(startTime: number | null): string {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!startTime) {
      setElapsed(0);
      return;
    }
    const tick = () => setElapsed(Math.floor((Date.now() - startTime) / 1000));
    tick();
    const id = setInterval(tick, 500);
    return () => clearInterval(id);
  }, [startTime]);

  const m = Math.floor(elapsed / 60);
  const s = elapsed % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

// Status dot color and animation per state
const DOT_CONFIG: Record<
  StatusKind,
  { color: string; glowColor: string; animated: boolean; pulse: boolean }
> = {
  disconnected: { color: 'var(--tahoe-text-tertiary)', glowColor: 'transparent', animated: false, pulse: false },
  connecting:   { color: 'var(--tahoe-warning)',       glowColor: 'transparent', animated: true,  pulse: true  },
  connected:    { color: 'var(--tahoe-success)',       glowColor: 'var(--tahoe-success)', animated: true, pulse: false },
  error:        { color: 'var(--tahoe-danger)',        glowColor: 'transparent', animated: false, pulse: false },
};

const TEXT_COLOR: Record<StatusKind, string> = {
  disconnected: 'var(--tahoe-text-tertiary)',
  connecting:   'var(--tahoe-warning)',
  connected:    'var(--tahoe-success)',
  error:        'var(--tahoe-danger)',
};

function StatusDot({ status }: { status: StatusKind }) {
  const cfg = DOT_CONFIG[status];
  return (
    <span
      className="inline-block w-2 h-2 rounded-full flex-shrink-0"
      style={{
        background: cfg.color,
        boxShadow: cfg.glowColor !== 'transparent'
          ? `0 0 6px 1px ${cfg.glowColor}`
          : 'none',
        animation: cfg.pulse
          ? 'tahoe-dot-pulse 1.2s ease-in-out infinite'
          : cfg.animated
          ? 'tahoe-glow-breathe 2s ease-in-out infinite'
          : 'none',
      }}
      aria-hidden="true"
    />
  );
}

function StatChip({ label, value }: { label: string; value: string | number }) {
  return (
    <div
      className="flex items-center gap-1.5 px-2.5 py-1 rounded-[var(--tahoe-radius-sm)]"
      style={{
        background: 'var(--tahoe-bg-glass)',
        border: '1px solid var(--tahoe-border-glass)',
      }}
    >
      <span
        className="text-[9px] font-semibold tracking-widest uppercase"
        style={{ color: 'var(--tahoe-text-tertiary)' }}
      >
        {label}
      </span>
      <span
        className="text-[11px] tabular-nums"
        style={{
          fontFamily: 'var(--tahoe-font-mono)',
          color: 'var(--tahoe-text-secondary)',
        }}
      >
        {value}
      </span>
    </div>
  );
}

export function SessionStatus({
  connected,
  status,
  statusText,
  stats,
  startTime,
}: SessionStatusProps) {
  const duration = useDuration(startTime);

  return (
    <>
      {/* Inline keyframes — injected once, scoped to component usage */}
      <style>{`
        @keyframes tahoe-dot-pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.55; transform: scale(0.8); }
        }
        @keyframes tahoe-glow-breathe {
          0%, 100% { box-shadow: 0 0 4px 1px var(--tahoe-success); }
          50% { box-shadow: 0 0 10px 3px var(--tahoe-success); }
        }
      `}</style>

      <GlassPanel
        as="div"
        className="px-4 py-2.5 flex items-center gap-3 flex-wrap"
        role="status"
        aria-live="polite"
        aria-label="Session status"
      >
        {/* Status indicator */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <StatusDot status={status} />
          <span
            className="text-xs font-medium"
            style={{
              color: TEXT_COLOR[status],
              transition: `color var(--tahoe-transition-base)`,
            }}
          >
            {statusText}
          </span>
        </div>

        {/* Vertical divider */}
        <div
          className="hidden sm:block w-px h-4 flex-shrink-0"
          style={{ background: 'var(--tahoe-border-glass)' }}
          aria-hidden="true"
        />

        {/* Stats row — pushed right */}
        <div className="flex items-center gap-2 flex-wrap ml-auto">
          <StatChip label="Time" value={duration} />
          <StatChip label="Sent" value={stats.framesSent} />
          <StatChip label="Recv" value={stats.framesRecv} />
          <StatChip label="Chars" value={stats.textChars} />
        </div>

        {/* Vertical divider */}
        <div
          className="hidden sm:block w-px h-4 flex-shrink-0"
          style={{ background: 'var(--tahoe-border-glass)' }}
          aria-hidden="true"
        />

        {/* Theme toggle */}
        <ThemeToggle />
      </GlassPanel>
    </>
  );
}
