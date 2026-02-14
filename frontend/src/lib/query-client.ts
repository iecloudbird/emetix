/**
 * TanStack Query Client Configuration
 *
 * Centralized query client with sensible defaults for financial data.
 */
import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes - stock data doesn't change rapidly
      gcTime: 30 * 60 * 1000, // 30 minutes cache
      retry: 2, // Retry twice on failure
      refetchOnWindowFocus: false, // Don't refetch when switching tabs
    },
  },
});
