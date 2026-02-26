"use client";

import { useEffect, useState, useCallback } from "react";
import { ServerCrash, Loader2, CheckCircle2, X } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const POLL_INTERVAL = 5000; // 5 seconds
const MAX_POLL_TIME = 180000; // 3 minutes

/**
 * Banner shown only in production when the backend API is unreachable
 * (Render free tier cold-starts). Auto-dismisses once the API responds.
 */
export function ColdStartBanner() {
  const [status, setStatus] = useState<
    "checking" | "down" | "up" | "dismissed"
  >(() => (process.env.NODE_ENV !== "production" ? "dismissed" : "checking"));
  const [elapsed, setElapsed] = useState(0);

  const checkHealth = useCallback(async () => {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 4000);
      const res = await fetch(`${API_URL}/health`, {
        signal: controller.signal,
      });
      clearTimeout(timeout);
      if (res.ok) return true;
    } catch {
      // Network error or timeout — API not ready
    }
    return false;
  }, []);

  useEffect(() => {
    // Only run on production builds
    if (process.env.NODE_ENV !== "production") return;

    let mounted = true;
    const startTime = Date.now();

    const poll = async () => {
      const healthy = await checkHealth();
      if (!mounted) return;
      if (healthy) {
        setStatus("up");
        clearInterval(pollTimer);
        clearInterval(elapsedTimer);
        // Auto-hide success state after 3s
        setTimeout(() => mounted && setStatus("dismissed"), 3000);
      } else {
        setStatus("down");
        if (Date.now() - startTime > MAX_POLL_TIME) {
          clearInterval(pollTimer);
          clearInterval(elapsedTimer);
        }
      }
    };

    // Initial check
    poll();

    // Poll at interval
    const pollTimer = setInterval(poll, POLL_INTERVAL);
    const elapsedTimer = setInterval(() => {
      if (mounted) setElapsed(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    return () => {
      mounted = false;
      clearInterval(pollTimer);
      clearInterval(elapsedTimer);
    };
  }, [checkHealth]);

  if (status === "dismissed" || status === "checking") return null;

  if (status === "up") {
    return (
      <div className="bg-green-500/10 border-b border-green-500/20 text-green-700 dark:text-green-400">
        <div className="container mx-auto px-4 py-2 flex items-center justify-center gap-2 text-sm">
          <CheckCircle2 className="h-4 w-4 shrink-0" />
          <span className="font-medium">
            Backend is online — you&apos;re all set!
          </span>
        </div>
      </div>
    );
  }

  // status === "down"
  return (
    <div className="bg-amber-500/10 border-b border-amber-500/20 text-amber-800 dark:text-amber-300">
      <div className="container mx-auto px-4 py-2.5 flex items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex items-center gap-2 shrink-0">
            <ServerCrash className="h-4 w-4" />
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          </div>
          <div className="text-sm leading-snug">
            <span className="font-semibold">Backend is waking up...</span>{" "}
            <span className="text-amber-700/80 dark:text-amber-400/80">
              This is a student FYP project hosted on Render&apos;s free tier —
              the server sleeps when idle and takes ~1-2 min to start.{" "}
              {elapsed > 0 && (
                <span className="tabular-nums">({elapsed}s)</span>
              )}
            </span>
          </div>
        </div>
        <button
          onClick={() => setStatus("dismissed")}
          className="shrink-0 p-1 rounded-md hover:bg-amber-500/20 transition-colors"
          aria-label="Dismiss"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}
