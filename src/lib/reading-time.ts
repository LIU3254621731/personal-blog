export function getReadingTime(content: string): number {
  const wordsPerMinute = 300; // Chinese characters per minute
  const chars = content.replace(/\s/g, "").length;
  return Math.max(1, Math.ceil(chars / wordsPerMinute));
}
