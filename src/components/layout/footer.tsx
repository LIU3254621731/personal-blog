import { siteConfig } from "@/data/site";

export function Footer() {
  return (
    <footer className="mt-32 pb-10">
      <div className="mx-auto max-w-3xl px-6">
        <div className="border-t border-border-light pt-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p className="text-xs text-text-tertiary tracking-wider">
            &copy; {new Date().getFullYear()} {siteConfig.author}
          </p>
          <div className="flex items-center gap-5">
            {siteConfig.nav.map((item) => (
              <a
                key={item.href}
                href={item.href}
                className="text-xs text-text-tertiary hover:text-text-secondary transition-colors tracking-wider"
              >
                {item.label}
              </a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}
