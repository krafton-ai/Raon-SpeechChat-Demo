'use client';

import { useRef, useEffect, useState } from 'react';
import { GlassPanel } from '@/components/ui/GlassPanel';

interface TranscriptPanelProps {
  text: string;
  seqTrace: string;
}

function EmptyState({ children }: { children: string }) {
  return (
    <p
      className="text-sm italic select-none"
      style={{
        color: 'var(--tahoe-text-tertiary)',
        fontFamily: 'var(--tahoe-font-mono)',
      }}
    >
      {children}
    </p>
  );
}

/** Filter <|...|> special tokens: replace with a single space */
function filterSpecialTokens(raw: string): string {
  return raw
    .replace(/<\|[^|]*\|>/g, ' ')     // Replace <|...|> tokens with space
    .replace(/ {2,}/g, ' ')            // Collapse multiple spaces to one
    .trim();
}

export function TranscriptPanel({ text, seqTrace }: TranscriptPanelProps) {
  const [traceOpen, setTraceOpen] = useState(false);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const traceRef = useRef<HTMLDivElement>(null);

  // Auto-scroll transcript to bottom when text changes
  useEffect(() => {
    const el = transcriptRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [text]);

  // Auto-scroll trace to bottom when trace changes
  useEffect(() => {
    const el = traceRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [seqTrace]);

  return (
    <GlassPanel className="flex flex-col overflow-hidden min-h-0 flex-1">
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 flex-shrink-0"
        style={{ borderBottom: '1px solid var(--tahoe-border-glass)' }}
      >
        <div className="flex items-center gap-2">
          <svg
            className="w-3.5 h-3.5 flex-shrink-0"
            style={{ color: 'var(--tahoe-text-tertiary)' }}
            viewBox="0 0 16 16"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M2 2.5A.5.5 0 0 1 2.5 2h11a.5.5 0 0 1 0 1h-11A.5.5 0 0 1 2 2.5zm0 4A.5.5 0 0 1 2.5 6h11a.5.5 0 0 1 0 1h-11A.5.5 0 0 1 2 6.5zm0 4A.5.5 0 0 1 2.5 10h6a.5.5 0 0 1 0 1h-6A.5.5 0 0 1 2 10.5z" />
          </svg>
          <span
            className="text-xs font-semibold tracking-wide"
            style={{ color: 'var(--tahoe-text-secondary)' }}
          >
            Transcript
          </span>
          {text && (
            <span
              className="text-[10px] tabular-nums"
              style={{
                fontFamily: 'var(--tahoe-font-mono)',
                color: 'var(--tahoe-text-tertiary)',
              }}
            >
              {text.length} chars
            </span>
          )}
        </div>

        {/* Sequence trace toggle */}
        <button
          onClick={() => setTraceOpen((v) => !v)}
          aria-expanded={traceOpen}
          aria-controls="seq-trace-panel"
          className={[
            'flex items-center gap-1.5 text-[10px] font-semibold tracking-widest uppercase',
            'px-2 py-1 rounded-[var(--tahoe-radius-sm)]',
            'transition-all',
          ].join(' ')}
          style={{
            transitionDuration: 'var(--tahoe-transition-fast)',
            background: traceOpen ? 'var(--tahoe-accent-light)' : 'transparent',
            color: traceOpen ? 'var(--tahoe-accent)' : 'var(--tahoe-text-tertiary)',
            border: traceOpen
              ? '1px solid var(--tahoe-accent)'
              : '1px solid transparent',
          }}
        >
          <svg
            className={['w-3 h-3 transition-transform', traceOpen ? 'rotate-180' : ''].join(' ')}
            style={{ transitionDuration: 'var(--tahoe-transition-base)' }}
            viewBox="0 0 12 12"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M6 8.5L1 3.5h10L6 8.5z" />
          </svg>
          Seq Trace
        </button>
      </div>

      {/* Transcript body */}
      <div
        ref={transcriptRef}
        className="flex-1 overflow-y-auto px-4 py-4 text-sm leading-relaxed whitespace-pre-wrap break-words min-h-0"
        style={{
          color: 'var(--tahoe-text-primary)',
          fontFamily: 'var(--tahoe-font-mono)',
          scrollBehavior: 'smooth',
        }}
        aria-live="polite"
        aria-label="Model transcript output"
      >
        {text ? (
          filterSpecialTokens(text)
        ) : (
          <div className="flex items-center justify-center h-full min-h-[120px]">
            <EmptyState>Waiting for model output…</EmptyState>
          </div>
        )}
      </div>

      {/* Sequence trace (collapsible) */}
      {traceOpen && (
        <div
          id="seq-trace-panel"
          className="flex flex-col flex-shrink-0"
          style={{
            maxHeight: '220px',
            borderTop: '1px solid var(--tahoe-border-glass)',
          }}
        >
          {/* Trace sub-header */}
          <div
            className="flex items-center justify-between px-4 py-2 flex-shrink-0"
            style={{ background: 'var(--tahoe-bg-secondary)' }}
          >
            <span
              className="tahoe-label"
              style={{ color: 'var(--tahoe-text-tertiary)' }}
            >
              Sequence Trace
            </span>
            {seqTrace && (
              <span
                className="text-[10px] tabular-nums"
                style={{
                  fontFamily: 'var(--tahoe-font-mono)',
                  color: 'var(--tahoe-text-tertiary)',
                }}
              >
                {seqTrace.split('\n').length} lines
              </span>
            )}
          </div>

          {/* Trace body */}
          <div
            ref={traceRef}
            className="overflow-auto px-4 py-3 text-[11px] leading-snug whitespace-pre flex-1"
            style={{
              fontFamily: 'var(--tahoe-font-mono)',
              color: 'var(--tahoe-text-secondary)',
            }}
            aria-label="Sequence trace"
          >
            {seqTrace ? (
              seqTrace
            ) : (
              <EmptyState>Waiting for sequence trace…</EmptyState>
            )}
          </div>
        </div>
      )}
    </GlassPanel>
  );
}
