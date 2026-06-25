"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Lock, X, Loader2 } from "lucide-react";

interface Props {
  open: boolean;
  onClose: () => void;
}

export function AdminLoginModal({ open, onClose }: Props) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!password.trim()) {
      setError("请输入密码");
      return;
    }
    setLoading(true);
    setError("");

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (res.ok) {
        router.push("/admin");
      } else {
        const data = await res.json();
        setError(data.error || "密码错误");
      }
    } catch {
      setError("网络错误，请重试");
    }
    setLoading(false);
  }

  function handleClose() {
    setPassword("");
    setError("");
    setLoading(false);
    onClose();
  }

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-[60] bg-black/20 backdrop-blur-sm"
            onClick={handleClose}
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 8 }}
            transition={{ duration: 0.25, ease: [0.22, 0.61, 0.36, 1] }}
            className="fixed inset-0 z-[61] flex items-center justify-center p-4"
          >
            <div
              className="glass-strong rounded-2xl p-6 w-full max-w-sm relative"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Close button */}
              <button
                onClick={handleClose}
                className="absolute top-4 right-4 p-1.5 rounded-lg text-text-tertiary hover:text-text-primary hover:bg-black/3 dark:hover:bg-white/5 transition-all"
              >
                <X size={18} />
              </button>

              <form onSubmit={handleSubmit} className="space-y-5">
                {/* Header */}
                <div className="text-center space-y-2 pt-2">
                  <div className="mx-auto w-10 h-10 rounded-xl bg-accent-light dark:bg-accent-light/20 flex items-center justify-center">
                    <Lock size={19} className="text-accent" />
                  </div>
                  <h2 className="font-display text-lg font-semibold text-text-primary">
                    管理员登录
                  </h2>
                  <p className="text-xs text-text-tertiary">
                    请输入密码以访问后台
                  </p>
                </div>

                {/* Input */}
                <div className="space-y-2">
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => {
                      setPassword(e.target.value);
                      setError("");
                    }}
                    placeholder="密码"
                    className="w-full px-4 py-3 rounded-xl border border-border-medium bg-bg-primary/50 text-text-primary text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-all duration-200"
                    autoFocus
                    disabled={loading}
                  />

                  {error && (
                    <motion.p
                      initial={{ opacity: 0, y: -4 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="text-red-500 text-xs"
                    >
                      {error}
                    </motion.p>
                  )}
                </div>

                {/* Submit */}
                <button
                  type="submit"
                  disabled={loading}
                  className="w-full py-2.5 rounded-xl bg-accent text-white text-sm font-medium flex items-center justify-center gap-2 hover:bg-accent/90 active:scale-[0.98] transition-all duration-200 disabled:opacity-50"
                >
                  {loading && <Loader2 size={16} className="animate-spin" />}
                  {loading ? "验证中..." : "登 录"}
                </button>
              </form>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
