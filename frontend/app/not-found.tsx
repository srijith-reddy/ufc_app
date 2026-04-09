import Link from "next/link";

export default function NotFound() {
  return (
    <div className="px-6 py-20 lg:px-10">
      <div className="mx-auto max-w-3xl rounded-[28px] border border-white/10 bg-white/[0.03] p-8 shadow-halo">
        <h2 className="text-3xl font-semibold tracking-[-0.04em] text-white">Page not found</h2>
        <p className="mt-4 text-sm leading-7 text-white/62">
          The route you requested is not part of the current product surface.
        </p>
        <Link
          href="/"
          className="mt-6 inline-flex rounded-full border border-white/12 bg-white/[0.04] px-5 py-3 text-sm font-medium text-white transition hover:bg-white/[0.08]"
        >
          Return home
        </Link>
      </div>
    </div>
  );
}
