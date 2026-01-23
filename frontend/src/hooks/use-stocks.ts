/**
 * Custom Hooks for Data Fetching
 *
 * Each hook encapsulates TanStack Query logic for a specific domain.
 * Follows the principle of co-locating data fetching with usage.
 */
import { useQuery } from "@tanstack/react-query";
import {
  fetchWatchlist,
  fetchCategorizedWatchlist,
  fetchCuratedWatchlist,
  fetchStock,
  fetchCharts,
  fetchSectors,
  compareStocks,
  type WatchlistResponse,
  type CategorizedWatchlistResponse,
  type CuratedWatchlistResponse,
  type StockDetailResponse,
  type ChartResponse,
  type SectorData,
  type Stock,
} from "@/lib/api";

// Watchlist hook - fetches top undervalued stocks
export function useWatchlist(n = 10, consensus = true) {
  return useQuery<WatchlistResponse>({
    queryKey: ["watchlist", n, consensus],
    queryFn: () => fetchWatchlist(n, consensus),
  });
}

// Categorized watchlist hook - fetches undervalued, quality, and growth lists
export function useCategorizedWatchlist(n = 10, consensus = true) {
  return useQuery<CategorizedWatchlistResponse>({
    queryKey: ["watchlist", "categorized", n, consensus],
    queryFn: () => fetchCategorizedWatchlist(n, consensus),
  });
}

// Curated watchlist hook - fetches Stage 3 curated recommendations
export function useCuratedWatchlist(category?: string) {
  return useQuery<CuratedWatchlistResponse>({
    queryKey: ["curated", "watchlist", category],
    queryFn: () => fetchCuratedWatchlist(category),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// Individual stock hook
export function useStock(ticker: string) {
  return useQuery<StockDetailResponse>({
    queryKey: ["stock", ticker],
    queryFn: () => fetchStock(ticker),
    enabled: Boolean(ticker), // Only fetch if ticker is provided
  });
}

// Chart data hook
export function useCharts(ticker: string) {
  return useQuery<ChartResponse>({
    queryKey: ["charts", ticker],
    queryFn: () => fetchCharts(ticker),
    enabled: Boolean(ticker),
  });
}

// Sectors overview hook
export function useSectors() {
  return useQuery<{ sectors: SectorData[] }>({
    queryKey: ["sectors"],
    queryFn: fetchSectors,
  });
}

// Compare multiple stocks hook
export function useCompare(tickers: string[]) {
  return useQuery<{ stocks: Stock[] }>({
    queryKey: ["compare", tickers],
    queryFn: () => compareStocks(tickers),
    enabled: tickers.length > 0,
  });
}
