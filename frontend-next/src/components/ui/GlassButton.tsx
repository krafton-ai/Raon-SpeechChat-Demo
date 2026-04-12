'use client';

import React, { ButtonHTMLAttributes, forwardRef } from 'react';

type Variant = 'default' | 'accent' | 'danger';
type Size = 'sm' | 'md' | 'lg';

interface GlassButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

const variantStyles: Record<Variant, string> = {
  default:
    'bg-[var(--tahoe-bg-glass)] hover:bg-[var(--tahoe-bg-glass-hover)] active:bg-[var(--tahoe-bg-glass-active)] ' +
    'text-[var(--tahoe-text-primary)] border-[var(--tahoe-border-glass)]',
  accent:
    'bg-[var(--tahoe-accent-light)] hover:bg-[var(--tahoe-accent)] active:bg-[var(--tahoe-accent-hover)] ' +
    'text-[var(--tahoe-accent)] hover:text-white active:text-white border-[var(--tahoe-accent)]',
  danger:
    'bg-[rgba(255,59,48,0.1)] hover:bg-[var(--tahoe-danger)] active:bg-[rgba(255,59,48,0.9)] ' +
    'text-[var(--tahoe-danger)] hover:text-white active:text-white border-[var(--tahoe-danger)]',
};

const sizeStyles: Record<Size, string> = {
  sm: 'px-3 py-1.5 text-[var(--tahoe-text-sm)] rounded-[var(--tahoe-radius-sm)]',
  md: 'px-4 py-2   text-[var(--tahoe-text-base)] rounded-[var(--tahoe-radius-md)]',
  lg: 'px-6 py-2.5 text-[var(--tahoe-text-lg)] rounded-[var(--tahoe-radius-lg)]',
};

const GlassButton = forwardRef<HTMLButtonElement, GlassButtonProps>(
  (
    {
      variant = 'default',
      size = 'md',
      className = '',
      children,
      disabled,
      ...rest
    },
    ref,
  ) => {
    return (
      <button
        ref={ref}
        disabled={disabled}
        className={[
          'inline-flex items-center justify-center gap-2 font-medium select-none',
          'backdrop-blur-[var(--tahoe-blur-md)] border',
          'shadow-[var(--tahoe-shadow-sm)]',
          'transition-all duration-[var(--tahoe-transition-base)]',
          'hover:scale-[1.02] active:scale-[0.98]',
          'hover:shadow-[var(--tahoe-shadow-md)]',
          'disabled:opacity-40 disabled:pointer-events-none',
          variantStyles[variant],
          sizeStyles[size],
          className,
        ]
          .filter(Boolean)
          .join(' ')}
        {...rest}
      >
        {children}
      </button>
    );
  },
);

GlassButton.displayName = 'GlassButton';

export { GlassButton };
export type { GlassButtonProps };
