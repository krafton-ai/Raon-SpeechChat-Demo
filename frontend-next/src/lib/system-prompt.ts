import type { PromptLanguage, SystemPromptStyle, TurnMode } from '@/types/protocol';

export interface PersonaPreset {
  category: string;
  label: string;
  value: string;
}

export const PERSONA_PRESETS: PersonaPreset[] = [
  { category: 'General', label: 'Friendly assistant', value: 'a friendly and helpful assistant' },
  { category: 'Game', label: 'Game host', value: 'a game host who facilitates interactive speech games with the user' },
  { category: 'Scenario', label: 'Movie guide', value: 'a movie and entertainment guide' },
  { category: 'Scenario', label: 'Banking advisor', value: 'a personal banking advisor' },
  { category: 'Scenario', label: 'Fitness coach', value: 'a fitness coach' },
  { category: 'Scenario', label: 'Shopping assistant', value: 'an online shopping assistant' },
  { category: 'Scenario', label: 'Pet care advisor', value: 'a pet care advisor' },
  { category: 'Scenario', label: 'Healthcare scheduler', value: 'a healthcare scheduling assistant' },
  { category: 'Scenario', label: 'Real estate advisor', value: 'a real estate advisor' },
  { category: 'Scenario', label: 'Tech support', value: 'a tech support specialist' },
  { category: 'Scenario', label: 'Car rental assistant', value: 'a car rental and navigation assistant' },
  { category: 'Scenario', label: 'Event planner', value: 'an event planning assistant' },
  { category: 'Scenario', label: 'Food delivery assistant', value: 'a restaurant and food delivery assistant' },
  { category: 'Scenario', label: 'Language tutor', value: 'a language tutor' },
  { category: 'Scenario', label: 'Travel planner', value: 'a travel planning assistant' },
  { category: 'Scenario', label: 'Interview coach', value: 'a job interview coach' },
  { category: 'Scenario', label: 'Game NPC', value: 'a game NPC assistant' },
];

const SYSTEM_PROMPT_BASE: Record<PromptLanguage, string> = {
  eng: 'You are engaging in real-time conversation.',
  kor: '당신은 실시간 대화에 참여하고 있습니다.',
};

function cleanOptionalText(value: string | undefined): string | undefined {
  const stripped = value?.trim();
  return stripped ? stripped : undefined;
}

function ensureTerminalPunctuation(text: string): string {
  const stripped = text.trim();
  if (!stripped) return stripped;
  return /[.!?]$/.test(stripped) ? stripped : `${stripped}.`;
}

export function makePromptKey(language: PromptLanguage, turnMode: TurnMode): string {
  return `${language}:full_duplex:${turnMode}`;
}

export function buildSystemPromptPreview({
  language,
  style,
  persona,
  context,
  customSystemPrompt,
}: {
  language: PromptLanguage;
  style: SystemPromptStyle;
  persona?: string;
  context?: string;
  customSystemPrompt?: string;
}): string {
  const base = SYSTEM_PROMPT_BASE[language];
  const personaText = cleanOptionalText(persona);
  const contextText = cleanOptionalText(context);
  const customText = cleanOptionalText(customSystemPrompt);

  if (style === 'custom') {
    return customText ? ensureTerminalPunctuation(customText) : base;
  }
  if (style === 'generic') {
    return base;
  }
  if (style === 'raon') {
    return language === 'kor' ? `${base} 당신은 도움이 되는 어시스턴트입니다.` : `${base} You are a helpful assistant.`;
  }
  if (!personaText) {
    return language === 'kor' ? `${base} 당신은 도움이 되는 어시스턴트입니다.` : `${base} You are a helpful assistant.`;
  }

  const personaPrompt = language === 'kor'
    ? ensureTerminalPunctuation(`${base} 당신은 도움이 되는 어시스턴트이고, ${personaText}`)
    : ensureTerminalPunctuation(`${base} You are a helpful assistant, ${personaText}.`);

  if (style === 'raon_persona_context' && contextText) {
    return `${personaPrompt} ${ensureTerminalPunctuation(contextText)}`;
  }

  return personaPrompt;
}
