import React, { ElementType, ComponentPropsWithoutRef } from 'react';

type GlassPanelOwnProps<E extends ElementType = 'div'> = {
  as?: E;
  className?: string;
  children?: React.ReactNode;
};

type GlassPanelProps<E extends ElementType = 'div'> = GlassPanelOwnProps<E> &
  Omit<ComponentPropsWithoutRef<E>, keyof GlassPanelOwnProps<E>>;

function GlassPanel<E extends ElementType = 'div'>({
  as,
  className = '',
  children,
  ...rest
}: GlassPanelProps<E>) {
  const Tag = (as ?? 'div') as ElementType;

  return (
    <Tag
      className={[
        'bg-[var(--tahoe-bg-glass)]',
        'backdrop-blur-[24px]',
        'border border-[var(--tahoe-border-glass)]',
        'rounded-[var(--tahoe-radius-lg)]',
        'shadow-[var(--tahoe-shadow-glass)]',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
      {...rest}
    >
      {children}
    </Tag>
  );
}

export { GlassPanel };
export type { GlassPanelProps };
