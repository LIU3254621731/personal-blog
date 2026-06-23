import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-32 text-center">
      <p className="font-mono text-[11px] tracking-[0.2em] text-text-tertiary uppercase mb-6">404</p>
      <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight mb-6">页面不存在</h1>
      <p className="text-text-secondary mb-10 leading-relaxed max-w-md mx-auto">
        你访问的页面可能已被移除、链接失效，或者从未存在过。
      </p>
      <div className="flex items-center justify-center gap-4">
        <Link href="/" className="inline-flex items-center gap-2 px-6 py-3 glass rounded-xl text-sm text-text-primary hover:shadow-md transition-all duration-300">
          ← 返回首页
        </Link>
        <Link href="/blog" className="inline-flex items-center gap-2 px-6 py-3 rounded-xl text-sm text-text-secondary hover:text-text-primary transition-colors">
          浏览文章
        </Link>
      </div>
    </div>
  );
}
