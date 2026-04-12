'use client';

import React, { ButtonHTMLAttributes, forwardRef } from 'react';

type IconButtonSize = 'sm' | 'md' | 'lg';
type IconButtonVariant = 'ghost' | 'glass' | 'accent';

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  size?: IconButtonSize;
  variant?: IconButtonVariant;
  /** Accessible label (sets aria-label) */
  label: string;
}

const sizeStyles: Record<IconButtonSize, string> = {
  sm: 'w-7 h-7 text-sm',
  md: 'w-9 h-9 text-base',
  lg: 'w-11 h-11 text-lg',
};

const variantStyles: Record<IconButtonVariant, string> = {
  ghost:
    'bg-transparent hover:bg-[var(--tahoe-border-subtle)] active:bg-[var(--tahoe-border-glass)] ' +
    'text-[var(--tahoe-text-secondary)] hover:text-[var(--tahoe-text-primary)]',
  glass:
    'bg-[var(--tahoe-bg-glass)] hover:bg-[var(--tahoe-bg-glass-hover)] active:bg-[var(--tahoe-bg-glass-active)] ' +
    'text-[var(--tahoe-text-secondary)] hover:text-[var(--tahoe-text-primary)] ' +
    'border border-[var(--tahoe-border-glass)] shadow-[var(--tahoe-shadow-sm)] backdrop-blur-[var(--tahoe-blur-md)]',
  accent:
    'bg-[var(--tahoe-accent-light)] hover:bg-[var(--tahoe-accent)] active:bg-[var(--tahoe-accent-hover)] ' +
    'text-[var(--tahoe-accent)] hover:text-white active:text-white ' +
    'border border-[var(--tahoe-accent)]',
};

const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  (
    {
      size = 'md',
      variant = 'ghost',
      label,
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
        aria-label={label}
        disabled={disabled}
        className={[
          'inline-flex items-center justify-center shrink-0',
          'rounded-[var(--tahoe-radius-md)]',
          'transition-all duration-[var(--tahoe-transition-fast)]',
          'hover:scale-[1.05] active:scale-[0.95]',
          'disabled:opacity-40 disabled:pointer-events-none',
          sizeStyles[size],
          variantStyles[variant],
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

IconButton.displayName = 'IconButton';

export { IconButton };
export type { IconButtonProps, IconButtonSize, IconButtonVariant };
