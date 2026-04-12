import React from 'react';

interface GlassCardProps {
  children?: React.ReactNode;
  className?: string;
  hover?: boolean;
  onClick?: React.MouseEventHandler<HTMLDivElement>;
}

function GlassCard({
  children,
  className = '',
  hover = false,
  onClick,
}: GlassCardProps) {
  const isInteractive = hover || !!onClick;

  return (
    <div
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') onClick(e as never);
            }
          : undefined
      }
      className={[
        'bg-[var(--tahoe-bg-glass)]',
        'backdrop-blur-[16px]',
        'border border-[var(--tahoe-border-glass)]',
        'rounded-[var(--tahoe-radius-md)]',
        'shadow-[var(--tahoe-shadow-sm)]',
        'transition-all duration-[var(--tahoe-transition-base)]',
        isInteractive &&
          'hover:bg-[var(--tahoe-bg-glass-hover)] hover:shadow-[var(--tahoe-shadow-md)] hover:scale-[1.01] cursor-pointer',
        onClick && 'focus-visible:outline-[var(--tahoe-accent)] focus-visible:outline-2 focus-visible:outline-offset-2',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {children}
    </div>
  );
}

export { GlassCard };
export type { GlassCardProps };
