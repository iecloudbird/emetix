/**
 * Pipeline Hooks - Phase 3 Quality Screening Pipeline
 *
 * Hooks for the 3-stage automated screening pipeline:
 * - Stage 1: Attention stocks (triggered by 52W Drop, Quality Growth, Deep Value)
 * - Stage 2: Qualified stocks (4-pillar scoring, composite >= 60)
 * - Stage 3: Classified stocks (Buy/Hold/Watch based on MoS + Score)
 */
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchPipelineAttention,
  fetchPipelineQualified,
  fetchPipelineClassified,
  fetchPipelineStock,
  fetchPipelineSummary,
  triggerPipelineScan,
  type PipelineAttentionResponse,
  type PipelineQualifiedResponse,
  type PipelineClassifiedResponse,
  type PipelineStockResponse,
  type PipelineSummaryResponse,
} from "@/lib/api";

// Attention stocks hook (Stage 1)
export function usePipelineAttention(
  trigger?: string,
  status: string = "active"
) {
  return useQuery<PipelineAttentionResponse>({
    queryKey: ["pipeline", "attention", trigger, status],
    queryFn: () => fetchPipelineAttention(trigger, status),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// Qualified stocks hook (Stage 2)
export function usePipelineQualified(
  classification?: "buy" | "hold" | "watch",
  sector?: string,
  minScore: number = 60
) {
  return useQuery<PipelineQualifiedResponse>({
    queryKey: ["pipeline", "qualified", classification, sector, minScore],
    queryFn: () => fetchPipelineQualified(classification, sector, minScore),
    staleTime: 5 * 60 * 1000,
  });
}

// Classified stocks hook (Stage 3 - Buy/Hold/Watch lists)
export function usePipelineClassified() {
  return useQuery<PipelineClassifiedResponse>({
    queryKey: ["pipeline", "classified"],
    queryFn: fetchPipelineClassified,
    staleTime: 5 * 60 * 1000,
  });
}

// Single stock in pipeline
export function usePipelineStock(ticker: string) {
  return useQuery<PipelineStockResponse>({
    queryKey: ["pipeline", "stock", ticker],
    queryFn: () => fetchPipelineStock(ticker),
    enabled: Boolean(ticker),
  });
}

// Pipeline summary hook
export function usePipelineSummary() {
  return useQuery<PipelineSummaryResponse>({
    queryKey: ["pipeline", "summary"],
    queryFn: fetchPipelineSummary,
    staleTime: 1 * 60 * 1000, // 1 minute
  });
}

// Trigger scan mutation
export function useTriggerScan() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      scanType,
      tickers,
    }: {
      scanType: "attention" | "qualified";
      tickers?: string[];
    }) => triggerPipelineScan(scanType, tickers),
    onSuccess: () => {
      // Invalidate pipeline queries after scan
      queryClient.invalidateQueries({ queryKey: ["pipeline"] });
    },
  });
}

// AI Analysis hooks
import {
  fetchAIAnalysis,
  fetchQuickAnalysis,
  type AIAnalysisResponse,
  type QuickAnalysisResponse,
} from "@/lib/api";

export function useAIAnalysis(
  ticker: string,
  options?: {
    includeEducation?: boolean;
    includeDiagnosis?: boolean;
    includeThesis?: boolean;
  }
) {
  return useQuery<AIAnalysisResponse>({
    queryKey: ["ai-analysis", ticker, options],
    queryFn: () => fetchAIAnalysis(ticker, options),
    enabled: Boolean(ticker),
    staleTime: 10 * 60 * 1000, // 10 minutes - AI analysis doesn't change often
  });
}

export function useQuickAnalysis(ticker: string) {
  return useQuery<QuickAnalysisResponse>({
    queryKey: ["quick-analysis", ticker],
    queryFn: () => fetchQuickAnalysis(ticker),
    enabled: Boolean(ticker),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

// Helper hook to get pillar score color
export function getPillarColor(score: number): string {
  if (score >= 75) return "text-green-600";
  if (score >= 50) return "text-yellow-600";
  return "text-red-600";
}

// Helper hook to get classification badge color
export function getClassificationColor(classification: string): {
  bg: string;
  text: string;
} {
  switch (classification) {
    case "buy":
      return { bg: "bg-green-100", text: "text-green-800" };
    case "hold":
      return { bg: "bg-blue-100", text: "text-blue-800" };
    case "watch":
      return { bg: "bg-yellow-100", text: "text-yellow-800" };
    default:
      return { bg: "bg-gray-100", text: "text-gray-800" };
  }
}

// Helper to format pillar name
export function formatPillarName(pillar: string): string {
  return pillar.charAt(0).toUpperCase() + pillar.slice(1);
}
