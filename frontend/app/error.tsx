"use client";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="px-6 py-20 lg:px-10">
      <div className="mx-auto max-w-3xl rounded-[28px] border border-rose-400/20 bg-rose-400/10 p-8">
        <h2 className="text-2xl font-semibold tracking-[-0.04em] text-white">
          Something interrupted the product surface
        </h2>
        <p className="mt-4 text-sm leading-7 text-white/72">
          {error.message || "Unexpected frontend error."}
        </p>
        <button
          type="button"
          onClick={reset}
          className="mt-6 rounded-full bg-white px-5 py-3 text-sm font-semibold text-[#05090e]"
        >
          Try again
        </button>
      </div>
    </div>
  );
}
