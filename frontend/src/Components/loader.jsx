// Loading states. `Loader` is the general spinner; the Skeleton* components
// mimic the shape of the content that's about to arrive, which reads as much
// faster than a bare spinner.

const Loader = ({ label = "Loading…", sub }) => (
  <div
    role="status"
    aria-live="polite"
    className="flex flex-col items-center justify-center gap-3 py-4"
  >
    <span className="relative inline-flex h-10 w-10">
      <span className="absolute inset-0 rounded-full border-[3px] border-slate-200 dark:border-slate-700" />
      <span className="absolute inset-0 animate-spin rounded-full border-[3px] border-transparent border-t-customGreen-100" />
    </span>
    <span className="iv-muted text-sm font-medium">{label}</span>
    {sub && <span className="iv-subtle max-w-md text-center text-xs">{sub}</span>}
  </div>
);

// A determinate variant for the backtest, which reports real progress.
export const ProgressLoader = ({ label, sub, progress }) => {
  const pct = Math.max(
    0,
    Math.min(100, Math.round((Number(progress) || 0) * 100))
  );
  return (
    <div role="status" aria-live="polite" className="mx-auto max-w-md py-4">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <span className="iv-muted text-sm font-medium">{label}</span>
        <span className="text-sm font-semibold text-customGreen-100">{pct}%</span>
      </div>
      <div
        className="iv-surface-muted h-2 w-full overflow-hidden rounded-full"
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className="h-full rounded-full bg-customGreen-100 transition-[width] duration-500 ease-out"
          style={{ width: `${Math.max(pct, 3)}%` }}
        />
      </div>
      {sub && <p className="iv-subtle mt-3 text-center text-xs">{sub}</p>}
    </div>
  );
};

export const SkeletonLine = ({ className = "h-4 w-full" }) => (
  <div className={`iv-skeleton ${className}`} />
);

export const SkeletonStats = ({ count = 4 }) => (
  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
    {Array.from({ length: count }).map((_, i) => (
      <div key={i} className="iv-card p-5">
        <SkeletonLine className="h-3 w-24" />
        <SkeletonLine className="mt-3 h-7 w-32" />
      </div>
    ))}
  </div>
);

export const SkeletonChart = ({ height = "h-[360px]" }) => (
  <div className="iv-card p-6">
    <SkeletonLine className="h-5 w-48" />
    <SkeletonLine className="mt-2 h-3 w-72 max-w-full" />
    <div className={`iv-skeleton mt-5 w-full ${height}`} />
  </div>
);

export const SkeletonTable = ({ rows = 6 }) => (
  <div className="iv-card overflow-hidden">
    <div className="iv-surface-muted h-12 w-full" />
    <div className="divide-y divide-[color:var(--iv-border)]">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex items-center gap-4 px-4 py-3.5">
          <SkeletonLine className="h-4 w-1/4" />
          <SkeletonLine className="h-4 flex-1" />
          <SkeletonLine className="hidden h-4 w-20 sm:block" />
          <SkeletonLine className="hidden h-4 w-20 md:block" />
        </div>
      ))}
    </div>
  </div>
);

export default Loader;
