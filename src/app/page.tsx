import { Hero } from "@/components/home/hero";
import { DailyStatus } from "@/components/home/daily-status";
import { FeaturedProjects } from "@/components/home/featured-projects";
import { LatestPosts } from "@/components/home/latest-posts";
import { Heatmap } from "@/components/home/heatmap";
import { Stats } from "@/components/home/stats";

export default function Home() {
  return (
    <div className="mx-auto max-w-5xl px-6">
      <Hero />

      {/* Two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-10">
        {/* Left: Main content */}
        <div className="min-w-0">
          <LatestPosts />
          <Heatmap />
        </div>

        {/* Right: Sidebar */}
        <aside className="space-y-10">
          <DailyStatus />
          <FeaturedProjects />
          <Stats />
        </aside>
      </div>
    </div>
  );
}
