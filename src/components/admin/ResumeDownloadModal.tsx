"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ShieldCheck, RefreshCw, Download, X } from "lucide-react";

interface Props {
  open: boolean;
  onClose: () => void;
  resumeUrl?: string;
}

/** Generate a simple math CAPTCHA */
function generateCaptcha() {
  const a = Math.floor(Math.random() * 20) + 1;
  const b = Math.floor(Math.random() * 20) + 1;
  const ops = ["+", "-", "×"] as const;
  const op = ops[Math.floor(Math.random() * ops.length)];
  let answer: number;
  let display: string;
  switch (op) {
    case "+":
      answer = a + b;
      display = `${a} + ${b} = ?`;
      break;
    case "-":
      answer = Math.max(a, b) - Math.min(a, b);
      display = `${Math.max(a, b)} - ${Math.min(a, b)} = ?`;
      break;
    case "×":
      answer = a * b;
      display = `${a} × ${b} = ?`;
      break;
    default:
      answer = a + b;
      display = `${a} + ${b} = ?`;
  }
  return { display, answer };
}

export function ResumeDownloadModal({
  open,
  onClose,
  resumeUrl = "/resume.pdf",
}: Props) {
  const [captcha, setCaptcha] = useState(generateCaptcha);
  const [input, setInput] = useState("");
  const [error, setError] = useState("");
  const [verified, setVerified] = useState(false);

  // Reset on open
  useEffect(() => {
    if (open) {
      setCaptcha(generateCaptcha());
      setInput("");
      setError("");
      setVerified(false);
    }
  }, [open]);

  function handleRefresh() {
    setCaptcha(generateCaptcha());
    setInput("");
    setError("");
  }

  function handleVerify() {
    const num = parseInt(input, 10);
    if (isNaN(num)) {
      setError("请输入数字");
      return;
    }
    if (num === captcha.answer) {
      setVerified(true);
      // Auto-trigger download
      setTimeout(() => {
        const a = document.createElement("a");
        a.href = resumeUrl;
        a.download = "resume.pdf";
        a.click();
        onClose();
      }, 600);
    } else {
      setError("答案不正确，请重试");
      setInput("");
      setCaptcha(generateCaptcha());
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter") handleVerify();
    if (e.key === "Escape") onClose();
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
            className="fixed inset-0 z-[60] bg-black/20 backdrop-blur-sm"
            onClick={onClose}
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
              {/* Close */}
              <button
                onClick={onClose}
                className="absolute top-4 right-4 p-1.5 rounded-lg text-text-tertiary hover:text-text-primary hover:bg-black/3 dark:hover:bg-white/5 transition-all"
              >
                <X size={18} />
              </button>

              {!verified ? (
                <div className="space-y-5 pt-2">
                  {/* Header */}
                  <div className="text-center space-y-2">
                    <div className="mx-auto w-10 h-10 rounded-xl bg-accent-light dark:bg-accent-light/20 flex items-center justify-center">
                      <ShieldCheck size={19} className="text-accent" />
                    </div>
                    <h2 className="font-display text-lg font-semibold text-text-primary">
                      下载验证
                    </h2>
                    <p className="text-xs text-text-tertiary">
                      请完成验证后下载简历
                    </p>
                  </div>

                  {/* Captcha */}
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-3 mb-4">
                      <span className="text-2xl font-mono font-bold text-text-primary tracking-wider select-none">
                        {captcha.display}
                      </span>
                      <button
                        onClick={handleRefresh}
                        className="p-1.5 rounded-lg text-text-tertiary hover:text-accent hover:bg-accent-light/50 transition-all"
                        title="换一题"
                      >
                        <RefreshCw size={16} />
                      </button>
                    </div>

                    <input
                      type="text"
                      inputMode="numeric"
                      value={input}
                      onChange={(e) => {
                        setInput(e.target.value.replace(/\D/g, ""));
                        setError("");
                      }}
                      onKeyDown={handleKeyDown}
                      placeholder="输入答案"
                      className="w-full px-4 py-3 rounded-xl border border-border-medium bg-bg-primary/50 text-text-primary text-center text-sm placeholder:text-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/20 focus:border-accent transition-all duration-200"
                      autoFocus
                    />

                    {error && (
                      <motion.p
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-red-500 text-xs mt-2"
                      >
                        {error}
                      </motion.p>
                    )}
                  </div>

                  {/* Verify */}
                  <button
                    onClick={handleVerify}
                    disabled={!input}
                    className="w-full py-2.5 rounded-xl bg-accent text-white text-sm font-medium hover:bg-accent/90 active:scale-[0.98] transition-all duration-200 disabled:opacity-40"
                  >
                    验证并下载
                  </button>
                </div>
              ) : (
                <div className="text-center space-y-4 py-6">
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 300 }}
                    className="mx-auto w-12 h-12 rounded-full bg-green-100 dark:bg-green-500/15 flex items-center justify-center"
                  >
                    <Download size={22} className="text-green-600 dark:text-green-400" />
                  </motion.div>
                  <p className="text-sm text-text-secondary">
                    验证通过，正在下载...
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
