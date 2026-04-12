'use client';

import { useRef, useEffect } from 'react';
import { GlassPanel } from '@/components/ui/GlassPanel';

interface AudioMetersProps {
  inputRms: number;
  outputRms: number;
}

const METER_MAX = 0.25; // RMS=0.25 → 100%

function rmsToDb(rms: number): string {
  if (rms < 1e-9) return '--';
  return (20 * Math.log10(rms)).toFixed(1);
}

// Resolve a CSS variable from the document root (client-side only)
function getCssVar(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback;
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function drawMeter(
  canvas: HTMLCanvasElement,
  rms: number,
  colorStart: string,
  colorEnd: string,
  trackColor: string,
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  const pct = Math.min(1, rms / METER_MAX);

  ctx.clearRect(0, 0, w, h);

  // Background track
  ctx.fillStyle = trackColor;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(0, 0, w, h, 4);
  } else {
    ctx.rect(0, 0, w, h);
  }
  ctx.fill();

  // Level bar with gradient
  const barW = Math.round(pct * w);
  if (barW > 1) {
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, colorStart);
    grad.addColorStop(0.7, colorEnd);
    grad.addColorStop(1, '#ef4444');
    ctx.fillStyle = grad;
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(0, 0, barW, h, 4);
    } else {
      ctx.rect(0, 0, barW, h);
    }
    ctx.fill();
  }

  // Tick marks at 25%, 50%, 75%
  ctx.fillStyle = trackColor;
  for (const tick of [0.25, 0.5, 0.75]) {
    const x = Math.round(tick * w);
    ctx.fillRect(x, 0, 1, h);
  }
}

function MeterBar({
  label,
  rms,
  accentVar,
  accentEndVar,
}: {
  label: string;
  rms: number;
  accentVar: string;
  accentEndVar: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(rect.width * dpr)) {
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.scale(dpr, dpr);
    }

    const colorStart = getCssVar(accentVar, '#3b82f6');
    const colorEnd = getCssVar(accentEndVar, '#60a5fa');
    const trackColor = getCssVar('--tahoe-border-subtle', 'rgba(0,0,0,0.08)');

    drawMeter(canvas, rms, colorStart, colorEnd, trackColor);
  }, [rms, accentVar, accentEndVar]);

  const db = rmsToDb(rms);
  const pct = Math.min(100, (rms / METER_MAX) * 100);
  const isActive = rms > 1e-9;

  return (
    <div className="flex flex-col gap-1.5">
      {/* Label row */}
      <div className="flex items-center justify-between">
        <span className="tahoe-label">{label}</span>
        <span
          className="text-[11px] tabular-nums transition-colors"
          style={{
            fontFamily: 'var(--tahoe-font-mono)',
            color: isActive ? 'var(--tahoe-text-primary)' : 'var(--tahoe-text-tertiary)',
            transitionDuration: 'var(--tahoe-transition-fast)',
          }}
        >
          {db === '--' ? '--' : db}{' '}
          <span style={{ color: 'var(--tahoe-text-tertiary)', fontSize: '9px' }}>dBFS</span>
        </span>
      </div>

      {/* Canvas meter */}
      <div className="relative h-4 w-full">
        <canvas
          ref={canvasRef}
          className="w-full h-full rounded"
          style={{ display: 'block' }}
          aria-hidden="true"
        />
      </div>

      {/* Thin progress underline */}
      <div
        className="h-px rounded overflow-hidden"
        style={{ background: 'var(--tahoe-border-subtle)' }}
      >
        <div
          className="h-full rounded"
          style={{
            width: `${pct}%`,
            background: `var(${accentVar})`,
            transition: 'width 75ms linear',
          }}
        />
      </div>
    </div>
  );
}

export function AudioMeters({ inputRms, outputRms }: AudioMetersProps) {
  return (
    <GlassPanel className="p-4">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <svg
          className="w-3.5 h-3.5 flex-shrink-0"
          style={{ color: 'var(--tahoe-text-tertiary)' }}
          viewBox="0 0 16 16"
          fill="currentColor"
          aria-hidden="true"
        >
          <path d="M8 1a3 3 0 0 0-3 3v4a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3zM4.5 8a.5.5 0 0 0-1 0 4.5 4.5 0 0 0 4 4.472V14H6a.5.5 0 0 0 0 1h4a.5.5 0 0 0 0-1H8.5v-1.528A4.5 4.5 0 0 0 12.5 8a.5.5 0 0 0-1 0 3.5 3.5 0 0 1-7 0z" />
        </svg>
        <span
          className="text-xs font-semibold tracking-wide"
          style={{ color: 'var(--tahoe-text-secondary)' }}
        >
          Audio Levels
        </span>
      </div>

      <div className="grid grid-cols-2 gap-5">
        {/* Mic — accent (blue) */}
        <MeterBar
          label="Mic Input"
          rms={inputRms}
          accentVar="--tahoe-accent"
          accentEndVar="--tahoe-accent-hover"
        />
        {/* Output — success (green) */}
        <MeterBar
          label="Model Output"
          rms={outputRms}
          accentVar="--tahoe-success"
          accentEndVar="--tahoe-success"
        />
      </div>
    </GlassPanel>
  );
}
