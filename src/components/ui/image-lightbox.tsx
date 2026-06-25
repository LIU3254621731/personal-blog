"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";

export function useImageLightbox() {
  const [imgSrc, setImgSrc] = useState<string | null>(null);

  const open = useCallback((src: string) => {
    setImgSrc(src);
    document.body.style.overflow = "hidden";
  }, []);

  const close = useCallback(() => {
    setImgSrc(null);
    document.body.style.overflow = "";
  }, []);

  // Esc to close
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    if (imgSrc) document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [imgSrc, close]);

  return { imgSrc, open, close };
}

interface Props {
  src: string | null;
  onClose: () => void;
}

export function ImageLightbox({ src, onClose }: Props) {
  if (!src) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        className="fixed inset-0 z-[200] bg-black/70 backdrop-blur-md flex items-center justify-center p-4"
        onClick={onClose}
      >
        <button
          onClick={onClose}
          className="absolute top-6 right-6 p-2 rounded-full bg-white/10 text-white hover:bg-white/20 transition-colors"
          aria-label="关闭"
        >
          <X size={20} />
        </button>
        <motion.img
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.9 }}
          transition={{ duration: 0.25, ease: [0.22, 0.61, 0.36, 1] }}
          src={src}
          alt=""
          className="max-w-full max-h-[90vh] rounded-2xl"
          onClick={(e) => e.stopPropagation()}
        />
      </motion.div>
    </AnimatePresence>
  );
}
