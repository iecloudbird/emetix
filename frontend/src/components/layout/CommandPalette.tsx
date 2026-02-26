/**
 * Command Palette Component
 *
 * Cmd+K / Ctrl+K triggered search dialog for quick stock navigation.
 * Searches across all pipeline stocks by ticker and company name.
 */
"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ArrowRight } from "lucide-react";
import { usePipelineClassified } from "@/hooks/use-pipeline";
import { cn } from "@/lib/utils";

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const router = useRouter();
  const { data: classified } = usePipelineClassified();
  const [query, setQuery] = React.useState("");
  const [selectedIndex, setSelectedIndex] = React.useState(0);
  const inputRef = React.useRef<HTMLInputElement>(null);

  // Combine all stocks
  const allStocks = React.useMemo(() => {
    if (!classified?.classified) return [];
    return [
      ...(classified.classified.buy || []),
      ...(classified.classified.hold || []),
      ...(classified.classified.watch || []),
    ];
  }, [classified]);

  // Filter stocks by query
  const results = React.useMemo(() => {
    if (!query.trim()) return allStocks.slice(0, 8);
    const q = query.toLowerCase();
    return allStocks
      .filter(
        (s) =>
          s.ticker.toLowerCase().includes(q) ||
          s.company_name?.toLowerCase().includes(q),
      )
      .slice(0, 8);
  }, [query, allStocks]);

  // Reset on open
  React.useEffect(() => {
    if (open) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  // Reset selection when results change
  React.useEffect(() => {
    setSelectedIndex(0);
  }, [results]);

  const navigateTo = (ticker: string) => {
    onOpenChange(false);
    router.push(`/stock/${ticker}`);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && results[selectedIndex]) {
      e.preventDefault();
      navigateTo(results[selectedIndex].ticker);
    }
  };

  const getClassColor = (cls: string) => {
    switch (cls) {
      case "buy":
        return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400";
      case "hold":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400";
      case "watch":
        return "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400";
      default:
        return "";
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px] p-0 gap-0">
        <DialogHeader className="sr-only">
          <DialogTitle>Search Stocks</DialogTitle>
        </DialogHeader>

        {/* Search Input */}
        <div className="px-3 border-b border-border/40">
          <Input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search stocks by ticker or company..."
            className="border-0 focus-visible:ring-0 focus-visible:ring-offset-0 h-12"
          />
        </div>

        {/* Results */}
        <div className="max-h-[300px] overflow-y-auto p-2">
          {results.length === 0 ? (
            <div className="py-6 text-center text-sm text-muted-foreground">
              No stocks found for &ldquo;{query}&rdquo;
            </div>
          ) : (
            <div className="space-y-0.5">
              {!query.trim() && (
                <p className="px-2 py-1 text-xs text-muted-foreground">
                  Popular stocks
                </p>
              )}
              {results.map((stock, index) => (
                <button
                  key={stock.ticker}
                  onClick={() => navigateTo(stock.ticker)}
                  className={cn(
                    "flex items-center justify-between w-full rounded-md px-3 py-2 text-sm transition-colors",
                    index === selectedIndex
                      ? "bg-accent text-accent-foreground"
                      : "hover:bg-muted",
                  )}
                >
                  <div className="flex items-center gap-2">
                    <div className="text-left">
                      <span className="font-semibold">{stock.ticker}</span>
                      <span className="ml-2 text-muted-foreground truncate max-w-[200px] inline-block align-bottom">
                        {stock.company_name}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge
                      variant="secondary"
                      className={cn(
                        "text-[10px]",
                        getClassColor(stock.classification),
                      )}
                    >
                      {stock.classification.toUpperCase()}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {stock.composite_score?.toFixed(0)}
                    </span>
                    <ArrowRight className="h-3 w-3 text-muted-foreground" />
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Footer hint */}
        <div className="flex items-center justify-between border-t border-border/40 px-3 py-2 text-xs text-muted-foreground">
          <span>{allStocks.length} stocks indexed</span>
          <div className="flex items-center gap-2">
            <span>↑↓ Navigate</span>
            <span>↵ Open</span>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
