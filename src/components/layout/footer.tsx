import { siteConfig } from "@/data/site";
import { AdminEntry } from "@/components/admin/AdminEntry";

export function Footer() {
  const footerLinks = [
    ...siteConfig.nav,
    { label: "RSS", href: "/feed.xml" },
  ];

  return (
    <footer className="mt-32 pb-10">
      <div className="mx-auto max-w-5xl px-6">
        <div className="border-t border-border-light pt-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-text-tertiary tracking-wider">
            &copy; {new Date().getFullYear()} {siteConfig.name}. Built with
            curiosity.
          </p>
          <div className="flex items-center gap-5">
            {footerLinks.map((item) => (
              <a
                key={item.href}
                href={item.href}
                className="text-xs text-text-tertiary hover:text-accent transition-colors tracking-wider"
              >
                {item.label}
              </a>
            ))}
            <AdminEntry />
          </div>
        </div>
      </div>
    </footer>
  );
}
