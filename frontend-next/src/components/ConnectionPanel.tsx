'use client';

import { useState, useCallback } from 'react';
import type { PromptLanguage, SessionParams, SystemPromptStyle, TurnMode } from '@/types/protocol';
import type { AudioConstraints } from '@/lib/audio-capture';
import { GlassButton } from '@/components/ui/GlassButton';
import { GlassPanel } from '@/components/ui/GlassPanel';
import { buildSystemPromptPreview, makePromptKey, PERSONA_PRESETS } from '@/lib/system-prompt';

interface ConnectionPanelProps {
  onConnect: (url: string, params: Omit<SessionParams, 'speaker_mode'>) => void;
  onDisconnect: () => void;
  connected: boolean;
  inputGain?: number;
  onGainChange?: (gain: number) => void;
  micThreshold?: number;
  onMicThresholdChange?: (threshold: number) => void;
  audioConstraints?: AudioConstraints;
  onAudioConstraintsChange?: (constraints: AudioConstraints) => void;
}

function FieldLabel({ htmlFor, children }: { htmlFor: string; children: React.ReactNode }) {
  return (
    <label
      htmlFor={htmlFor}
      className="tahoe-label block mb-1.5"
    >
      {children}
    </label>
  );
}

function TextInput({
  id,
  value,
  onChange,
  placeholder,
  disabled,
  spellCheck = false,
  className = '',
}: {
  id: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  disabled?: boolean;
  spellCheck?: boolean;
  className?: string;
}) {
  return (
    <input
      id={id}
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
      spellCheck={spellCheck}
      className={[
        'tahoe-input w-full px-3 py-2 text-sm font-mono',
        'disabled:opacity-40 disabled:cursor-not-allowed',
        className,
      ].join(' ')}
    />
  );
}

function SelectInput({
  id,
  value,
  onChange,
  disabled,
  children,
}: {
  id: string;
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <select
      id={id}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      className="tahoe-input w-full px-3 py-2 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
      style={{ fontFamily: 'var(--tahoe-font-mono)' }}
    >
      {children}
    </select>
  );
}

export function ConnectionPanel({
  onConnect, onDisconnect, connected,
  inputGain = 1.0, onGainChange,
  micThreshold = 0, onMicThresholdChange,
  audioConstraints, onAudioConstraintsChange,
}: ConnectionPanelProps) {
  const defaultUrl =
    typeof window !== 'undefined'
      ? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}${window.location.pathname.replace(/\/+$/, '')}/ws/chat`
      : 'ws://localhost:8000/ws/chat';

  const [url, setUrl] = useState(defaultUrl);
  const [promptLanguage, setPromptLanguage] = useState<PromptLanguage>('eng');
  const [turnMode, setTurnMode] = useState<TurnMode>('listen-first');
  const [systemPromptStyle, setSystemPromptStyle] = useState<SystemPromptStyle>('raon_persona');
  const [personaPreset, setPersonaPreset] = useState(PERSONA_PRESETS[0]?.value ?? '');
  const [personaText, setPersonaText] = useState(PERSONA_PRESETS[0]?.value ?? '');
  const [contextText, setContextText] = useState('');
  const [customSystemPrompt, setCustomSystemPrompt] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [topKStr, setTopKStr] = useState('50');
  const [topPStr, setTopPStr] = useState('0.8');
  const [eosPenaltyStr, setEosPenaltyStr] = useState('0.0');
  const [bcPenaltyStr, setBcPenaltyStr] = useState('0.0');
  const [repPenaltyStr, setRepPenaltyStr] = useState('1.0');

  const prompt = makePromptKey(promptLanguage, turnMode);
  const previewPrompt = buildSystemPromptPreview({
    language: promptLanguage,
    style: systemPromptStyle,
    persona: personaText,
    context: contextText,
    customSystemPrompt,
  });

  const handleConnect = useCallback(() => {
    if (!url.trim()) return;
    onConnect(url.trim(), {
      prompt,
      prompt_language: promptLanguage,
      system_prompt_style: systemPromptStyle,
      system_prompt_persona:
        systemPromptStyle === 'raon_persona' || systemPromptStyle === 'raon_persona_context'
          ? personaText.trim()
          : undefined,
      system_prompt_context:
        systemPromptStyle === 'raon_persona_context'
          ? contextText.trim()
          : undefined,
      custom_system_prompt:
        systemPromptStyle === 'custom'
          ? customSystemPrompt.trim()
          : undefined,
      temperature,
      top_k: parseInt(topKStr) || 50,
      top_p: parseFloat(topPStr) || 0.8,
      eos_penalty: parseFloat(eosPenaltyStr) || 0,
      bc_penalty: parseFloat(bcPenaltyStr) || 0,
      repetition_penalty: parseFloat(repPenaltyStr) || 1.0,
    });
  }, [
    url,
    prompt,
    promptLanguage,
    systemPromptStyle,
    personaText,
    contextText,
    customSystemPrompt,
    temperature,
    topKStr,
    topPStr,
    eosPenaltyStr,
    bcPenaltyStr,
    repPenaltyStr,
    onConnect,
  ]);

  const handleToggle = useCallback(() => {
    if (connected) {
      onDisconnect();
    } else {
      handleConnect();
    }
  }, [connected, onDisconnect, handleConnect]);

  const tempPercent = ((temperature - 0.1) / (2.0 - 0.1)) * 100;

  return (
    <GlassPanel className="p-4 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <svg
          className="w-3.5 h-3.5 flex-shrink-0"
          style={{ color: 'var(--tahoe-accent)' }}
          viewBox="0 0 16 16"
          fill="currentColor"
          aria-hidden="true"
        >
          <path d="M8.354 1.146a.5.5 0 0 0-.708 0l-6 6A.5.5 0 0 0 1.5 7.5v7a.5.5 0 0 0 .5.5h4.5a.5.5 0 0 0 .5-.5v-4h2v4a.5.5 0 0 0 .5.5H14a.5.5 0 0 0 .5-.5v-7a.5.5 0 0 0-.146-.354L13 5.793V2.5a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0-.5.5v1.293L8.354 1.146z" />
        </svg>
        <span
          className="text-xs font-semibold tracking-wide"
          style={{ color: 'var(--tahoe-text-secondary)' }}
        >
          Connection
        </span>
      </div>

      {/* Server URL */}
      <div>
        <FieldLabel htmlFor="conn-url">Server URL</FieldLabel>
        <TextInput
          id="conn-url"
          value={url}
          onChange={setUrl}
          placeholder="ws://localhost:8000/ws/chat"
          disabled={connected}
        />
      </div>

      {/* Connect / Disconnect button */}
      <GlassButton
        variant={connected ? 'danger' : 'accent'}
        size="md"
        onClick={handleToggle}
        className="w-full justify-center font-semibold tracking-wide"
        aria-label={connected ? 'Disconnect from server' : 'Connect to server'}
      >
        {connected ? 'Disconnect' : 'Connect'}
      </GlassButton>

      {/* Turn Mode */}
      <div>
        <FieldLabel htmlFor="conn-language">Turn Mode</FieldLabel>
        <div className="grid grid-cols-2 gap-3 mb-3">
          <div>
            <div
              className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.14em]"
              style={{ color: 'var(--tahoe-text-tertiary)' }}
            >
              Language
            </div>
            <div className="flex gap-1.5">
              {([
                { label: 'English', value: 'eng' },
                { label: 'Korean', value: 'kor' },
              ] as const).map((option) => (
                <button
                  key={option.value}
                  onClick={() => setPromptLanguage(option.value)}
                  disabled={connected}
                  className="px-2 py-0.5 text-[10px] font-medium rounded-full transition-all"
                  style={{
                    background: promptLanguage === option.value ? 'var(--tahoe-accent-light)' : 'var(--tahoe-bg-glass)',
                    color: promptLanguage === option.value ? 'var(--tahoe-accent)' : 'var(--tahoe-text-secondary)',
                    border: promptLanguage === option.value ? '1px solid var(--tahoe-accent)' : '1px solid var(--tahoe-border-glass)',
                    opacity: connected ? 0.5 : 1,
                  }}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
          <div>
            <div
              className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.14em]"
              style={{ color: 'var(--tahoe-text-tertiary)' }}
            >
              Start Turn
            </div>
            <div className="flex gap-1.5">
              {([
                { label: 'Listen First', value: 'listen-first' },
                { label: 'Speak First', value: 'speak-first' },
              ] as const).map((option) => (
                <button
                  key={option.value}
                  onClick={() => setTurnMode(option.value)}
                  disabled={connected}
                  className="px-2 py-0.5 text-[10px] font-medium rounded-full transition-all"
                  style={{
                    background: turnMode === option.value ? 'var(--tahoe-accent-light)' : 'var(--tahoe-bg-glass)',
                    color: turnMode === option.value ? 'var(--tahoe-accent)' : 'var(--tahoe-text-secondary)',
                    border: turnMode === option.value ? '1px solid var(--tahoe-accent)' : '1px solid var(--tahoe-border-glass)',
                    opacity: connected ? 0.5 : 1,
                  }}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </div>
        <div
          className="text-[10px] font-mono"
          style={{ color: 'var(--tahoe-text-tertiary)' }}
        >
          Session key: {prompt}
        </div>
      </div>

      {/* System Prompt */}
      <div>
        <FieldLabel htmlFor="conn-prompt-style">System Prompt</FieldLabel>
        <div className="grid grid-cols-1 gap-3">
          <div>
            <SelectInput
              id="conn-prompt-style"
              value={systemPromptStyle}
              onChange={(v) => setSystemPromptStyle(v as SystemPromptStyle)}
              disabled={connected}
            >
              <option value="generic">[1] Generic</option>
              <option value="raon">[2] Assistant</option>
              <option value="raon_persona">[3] Assistant + persona</option>
              <option value="raon_persona_context">[4] Assistant + persona + context</option>
              <option value="custom">Custom prompt</option>
            </SelectInput>
          </div>
        </div>

        {(systemPromptStyle === 'raon_persona' || systemPromptStyle === 'raon_persona_context') && (
          <div className="grid grid-cols-1 gap-3 mt-3">
            <div>
              <FieldLabel htmlFor="conn-persona-preset">Persona Preset</FieldLabel>
              <SelectInput
                id="conn-persona-preset"
                value={personaPreset}
                onChange={(value) => {
                  setPersonaPreset(value);
                  if (value !== '__custom__') setPersonaText(value);
                }}
                disabled={connected}
              >
                {PERSONA_PRESETS.map((preset) => (
                  <option key={preset.value} value={preset.value}>
                    {preset.category} · {preset.label}
                  </option>
                ))}
                <option value="__custom__">Custom persona</option>
              </SelectInput>
            </div>
            <div>
              <FieldLabel htmlFor="conn-persona-text">Persona Text</FieldLabel>
              <textarea
                id="conn-persona-text"
                value={personaText}
                onChange={(e) => {
                  setPersonaPreset('__custom__');
                  setPersonaText(e.target.value);
                }}
                placeholder="a friendly and helpful assistant"
                disabled={connected}
                rows={3}
                className="tahoe-input w-full resize-y text-xs"
                style={{
                  fontFamily: 'var(--tahoe-font-mono)',
                  minHeight: '4rem',
                }}
              />
            </div>
          </div>
        )}

        {systemPromptStyle === 'raon_persona_context' && (
          <div className="mt-3">
            <FieldLabel htmlFor="conn-context-text">Context</FieldLabel>
            <textarea
              id="conn-context-text"
              value={contextText}
              onChange={(e) => setContextText(e.target.value)}
              placeholder="Optional scene, task, or conversation context"
              disabled={connected}
              rows={3}
              className="tahoe-input w-full resize-y text-xs"
              style={{
                fontFamily: 'var(--tahoe-font-mono)',
                minHeight: '4rem',
              }}
            />
          </div>
        )}

        {systemPromptStyle === 'custom' && (
          <div className="mt-3">
            <FieldLabel htmlFor="conn-custom-prompt">Custom System Prompt</FieldLabel>
            <textarea
              id="conn-custom-prompt"
              value={customSystemPrompt}
              onChange={(e) => setCustomSystemPrompt(e.target.value)}
              placeholder="You are engaging in real-time conversation. You are a friendly and helpful assistant."
              disabled={connected}
              rows={4}
              className="tahoe-input w-full resize-y text-xs"
              style={{
                fontFamily: 'var(--tahoe-font-mono)',
                minHeight: '5rem',
              }}
            />
          </div>
        )}

        <div className="mt-3">
          <div
            className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.14em]"
            style={{ color: 'var(--tahoe-text-tertiary)' }}
          >
            Preview
          </div>
          <textarea
            id="conn-prompt"
            value={previewPrompt}
            readOnly
            disabled
            rows={4}
            className="tahoe-input w-full resize-y text-xs opacity-80"
            style={{
              fontFamily: 'var(--tahoe-font-mono)',
              minHeight: '5rem',
            }}
          />
        </div>

        <div
          className="mt-4 pt-3"
          style={{ borderTop: '1px solid var(--tahoe-border-subtle)' }}
        >
          <div
            className="mb-1.5 text-[10px] font-semibold uppercase tracking-[0.14em]"
            style={{ color: 'var(--tahoe-text-tertiary)' }}
          >
            Prompt Format Reference
          </div>
          <div
            className="rounded-[var(--tahoe-radius-md)] px-3 py-2 space-y-1.5"
            style={{
              background: 'var(--tahoe-bg-secondary)',
              border: '1px solid var(--tahoe-border-subtle)',
            }}
          >
            {[
              '[1] You are engaging in real-time conversation.',
              '[2] You are engaging in real-time conversation. You are a helpful assistant.',
              '[3] You are engaging in real-time conversation. You are a helpful assistant, {persona}.',
              '[4] You are engaging in real-time conversation. You are a helpful assistant, {persona}. {context}.',
            ].map((line) => (
              <div
                key={line}
                className="text-[10px] leading-relaxed"
                style={{
                  color: 'var(--tahoe-text-secondary)',
                  fontFamily: 'var(--tahoe-font-mono)',
                }}
              >
                {line}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Temperature */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <FieldLabel htmlFor="conn-temp">Temperature</FieldLabel>
          <span
            className="text-xs tabular-nums"
            style={{
              fontFamily: 'var(--tahoe-font-mono)',
              color: 'var(--tahoe-accent)',
            }}
          >
            {temperature.toFixed(2)}
          </span>
        </div>
        <div className="relative">
          <input
            id="conn-temp"
            type="range"
            min={0.1}
            max={2.0}
            step={0.05}
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            disabled={connected}
            className="tahoe-range w-full"
            style={{
              background: `linear-gradient(to right, var(--tahoe-accent) ${tempPercent}%, var(--tahoe-border-subtle) ${tempPercent}%)`,
            }}
          />
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-[10px]" style={{ color: 'var(--tahoe-text-tertiary)' }}>0.1</span>
          <span className="text-[10px]" style={{ color: 'var(--tahoe-text-tertiary)' }}>2.0</span>
        </div>
      </div>

      {/* Top-K + Top-P side by side */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <FieldLabel htmlFor="conn-topk">Top-K</FieldLabel>
          <input
            id="conn-topk"
            type="text"
            inputMode="numeric"
            value={topKStr}
            onChange={(e) => setTopKStr(e.target.value)}
            disabled={connected}
            placeholder="50"
            className="tahoe-input w-full px-3 py-2 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ fontFamily: 'var(--tahoe-font-mono)' }}
          />
        </div>
        <div>
          <FieldLabel htmlFor="conn-topp">Top-P</FieldLabel>
          <input
            id="conn-topp"
            type="text"
            inputMode="decimal"
            value={topPStr}
            onChange={(e) => setTopPStr(e.target.value)}
            disabled={connected}
            placeholder="0.8"
            className="tahoe-input w-full px-3 py-2 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ fontFamily: 'var(--tahoe-font-mono)' }}
          />
        </div>
      </div>

      {/* EOS Penalty + Repetition Penalty side by side */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <FieldLabel htmlFor="conn-eos">EOS Penalty</FieldLabel>
          <input
            id="conn-eos"
            type="text"
            inputMode="decimal"
            value={eosPenaltyStr}
            onChange={(e) => setEosPenaltyStr(e.target.value)}
            disabled={connected}
            placeholder="0.0"
            className="tahoe-input w-full px-3 py-2 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ fontFamily: 'var(--tahoe-font-mono)' }}
          />
        </div>
        <div>
          <FieldLabel htmlFor="conn-bc">BC Penalty</FieldLabel>
          <input
            id="conn-bc"
            type="text"
            inputMode="decimal"
            value={bcPenaltyStr}
            onChange={(e) => setBcPenaltyStr(e.target.value)}
            disabled={connected}
            placeholder="0.0"
            className="tahoe-input w-full px-3 py-2 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ fontFamily: 'var(--tahoe-font-mono)' }}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3">
        <div>
          <FieldLabel htmlFor="conn-rep">Rep. Penalty</FieldLabel>
          <input
            id="conn-rep"
            type="text"
            inputMode="decimal"
            value={repPenaltyStr}
            onChange={(e) => setRepPenaltyStr(e.target.value)}
            disabled={connected}
            placeholder="1.0"
            className="tahoe-input w-full px-3 py-2 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
            style={{ fontFamily: 'var(--tahoe-font-mono)' }}
          />
        </div>
      </div>

      {/* Mic Gain — adjustable while connected */}
      {onGainChange && (
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <FieldLabel htmlFor="conn-gain">Mic Gain</FieldLabel>
            <span
              className="text-xs tabular-nums"
              style={{
                fontFamily: 'var(--tahoe-font-mono)',
                color: 'var(--tahoe-accent)',
              }}
            >
              {inputGain.toFixed(1)}x
            </span>
          </div>
          <div className="relative">
            <input
              id="conn-gain"
              type="range"
              min={0.0}
              max={2.0}
              step={0.1}
              value={inputGain}
              onChange={(e) => onGainChange(parseFloat(e.target.value))}
              className="tahoe-range w-full"
              style={{
                background: `linear-gradient(to right, var(--tahoe-accent) ${((inputGain) / 2.0) * 100}%, var(--tahoe-border-subtle) ${((inputGain) / 2.0) * 100}%)`,
              }}
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-[10px]" style={{ color: 'var(--tahoe-text-tertiary)' }}>0.0</span>
            <span className="text-[10px]" style={{ color: 'var(--tahoe-text-tertiary)' }}>1.0</span>
            <span className="text-[10px]" style={{ color: 'var(--tahoe-text-tertiary)' }}>2.0</span>
          </div>
        </div>
      )}

      {/* Noise Gate — adjustable while connected */}
      {onMicThresholdChange && (
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <FieldLabel htmlFor="conn-gate">Noise Gate</FieldLabel>
            <span
              className="text-xs tabular-nums"
              style={{
                fontFamily: 'var(--tahoe-font-mono)',
                color: micThreshold > 0 ? 'var(--tahoe-accent)' : 'var(--tahoe-text-tertiary)',
              }}
            >
              {micThreshold === 0 ? 'Off' : micThreshold.toFixed(3)}
            </span>
          </div>
          <div className="relative">
            <input
              id="conn-gate"
              type="range"
              min={0.0}
              max={0.05}
              step={0.001}
              value={micThreshold}
              onChange={(e) => onMicThresholdChange(parseFloat(e.target.value))}
              className="tahoe-range w-full"
              style={{
                background: `linear-gradient(to right, var(--tahoe-accent) ${(micThreshold / 0.05) * 100}%, var(--tahoe-border-subtle) ${(micThreshold / 0.05) * 100}%)`,
              }}
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-[10px]" style={{ color: 'var(--tahoe-text-tertiary)' }}>Off</span>
            <span className="text-[10px]" style={{ color: 'var(--tahoe-text-tertiary)' }}>0.05</span>
          </div>
        </div>
      )}

      {/* WebRTC Audio Processing */}
      {audioConstraints && onAudioConstraintsChange && (
        <div>
          <FieldLabel htmlFor="conn-webrtc">Audio Processing</FieldLabel>
          <div className="flex flex-wrap gap-1.5">
            {([
              { key: 'echoCancellation' as const, label: 'Echo Cancel' },
              { key: 'noiseSuppression' as const, label: 'Noise Suppress' },
              { key: 'autoGainControl' as const, label: 'Auto Gain' },
            ]).map(({ key, label }) => (
              <button
                key={key}
                onClick={() => onAudioConstraintsChange({ ...audioConstraints, [key]: !audioConstraints[key] })}
                disabled={connected}
                className="px-2 py-0.5 text-[10px] font-medium rounded-full transition-all"
                style={{
                  background: audioConstraints[key] ? 'var(--tahoe-accent-light)' : 'var(--tahoe-bg-glass)',
                  color: audioConstraints[key] ? 'var(--tahoe-accent)' : 'var(--tahoe-text-secondary)',
                  border: audioConstraints[key] ? '1px solid var(--tahoe-accent)' : '1px solid var(--tahoe-border-glass)',
                  opacity: connected ? 0.5 : 1,
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}

    </GlassPanel>
  );
}
