/**
 * API Client for Emetix Backend
 *
 * Centralized API configuration following service-oriented architecture.
 * All API calls go through this module for consistency and error handling.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Types for API responses

export interface ForwardMetrics {
  forward_pe: number | null;
  trailing_pe: number | null;
  pe_trend: "IMPROVING" | "STABLE" | "DECLINING";
  peg_ratio: number | null;
  earnings_growth: number;
  revenue_growth: number;
}

export type ValuationStatus =
  | "SIGNIFICANTLY_UNDERVALUED"
  | "MODERATELY_UNDERVALUED"
  | "SLIGHTLY_UNDERVALUED"
  | "FAIRLY_VALUED"
  | "SLIGHTLY_OVERVALUED"
  | "OVERVALUED";

export type ListCategory = "UNDERVALUED" | "QUALITY" | "GROWTH" | "GENERAL";

export interface Stock {
  rank?: number;
  ticker: string;
  company_name: string;
  sector: string;
  industry?: string;
  current_price: number;
  fair_value: number;
  margin_of_safety: number;
  margin_of_safety_raw?: number; // Uncapped actual value from backend
  market_cap?: number;
  market_cap_formatted?: string;
  pe_ratio: number | null;
  forward_pe?: number | null;
  pb_ratio: number | null;
  peg_ratio?: number | null;
  dividend_yield: number | null;
  beta: number | null;
  volatility?: number;
  valuation_score: number;
  effective_score?: number;
  consensus_score?: number;
  consensus_confidence?: number;
  lstm_fair_value: number | null;
  traditional_fair_value?: number | null;
  fair_value_method?: string;
  lstm_predicted_growth?: number | null;
  recommendation: string;
  assessment?: string;
  risk_level?: "LOW" | "MEDIUM" | "HIGH";
  justification: string;

  // New forward-looking fields
  valuation_status?: ValuationStatus;
  list_category?: ListCategory;
  forward_metrics?: ForwardMetrics;

  // Financial metrics
  roe?: number;
  roa?: number;
  debt_equity?: number;
  current_ratio?: number;
  profit_margin?: number;
  gross_margin?: number;
  fcf_yield?: number;
  revenue_growth?: number;
  earnings_growth?: number;

  // Performance
  pct_from_52w_high?: number;
  ytd_return?: number;
  analyst_target?: number;
  analyst_upside?: number;
}

export interface WatchlistResponse {
  status: string;
  timestamp?: string;
  model_info?: {
    lstm_enabled: boolean;
    lstm_model: string | null;
    rf_enabled: boolean;
    consensus_enabled: boolean;
    valuation_method: string;
  };
  summary?: {
    results_count: number;
    undervalued_count?: number;
    overvalued_count?: number;
    avg_valuation_score: number;
    avg_margin_of_safety: number;
    sector_distribution: Record<string, number>;
  };
  watchlist: Stock[];
  quality_list?: Stock[];
  growth_list?: Stock[];
  lists_explanation?: {
    undervalued: string;
    quality: string;
    growth: string;
  };
}

export interface CategorizedWatchlistResponse {
  status: string;
  timestamp: string;
  count_per_category: number;
  categories: {
    undervalued: string;
    quality: string;
    growth: string;
  };
  undervalued: Stock[];
  quality: Stock[];
  growth: Stock[];
}

export interface StockDetailResponse {
  status: string;
  data: Stock;
  sector_benchmarks?: {
    avg_pe: number;
    avg_pb: number;
    avg_roe: number;
  };
}

export interface ChartDataPoint {
  date: string;
  close?: number;
  price?: number; // Backend may return 'price' instead of 'close'
  volume?: number;
  ma50?: number | null;
  ma200?: number | null;
}

export interface ChartResponse {
  status: string;
  data: {
    ticker: string;
    company_name: string;
    charts: {
      price_1y: ChartDataPoint[];
      price_5y: ChartDataPoint[];
      technical: ChartDataPoint[];
    };
    metrics: {
      current_price: number;
      high_52w: number;
      low_52w: number;
      pe_ratio: number;
      market_cap: number;
    };
  };
}

export interface SectorData {
  sector: string;
  stock_count: number;
  avg_valuation_score: number;
  avg_upside: number;
}

// API Functions

export async function fetchWatchlist(
  n = 10,
  consensus = true,
): Promise<WatchlistResponse> {
  const res = await fetch(
    `${API_URL}/api/screener/watchlist?n=${n}&consensus=${consensus}`,
  );
  if (!res.ok) throw new Error(`Failed to fetch watchlist: ${res.status}`);
  return res.json();
}

export async function fetchCategorizedWatchlist(
  n = 10,
  consensus = true,
): Promise<CategorizedWatchlistResponse> {
  const res = await fetch(
    `${API_URL}/api/screener/watchlist/categorized?n=${n}&consensus=${consensus}`,
  );
  if (!res.ok)
    throw new Error(`Failed to fetch categorized watchlist: ${res.status}`);
  return res.json();
}

export async function fetchStock(ticker: string): Promise<StockDetailResponse> {
  const res = await fetch(
    `${API_URL}/api/screener/stock/${ticker.toUpperCase()}`,
  );
  if (!res.ok) throw new Error(`Failed to fetch stock: ${res.status}`);
  return res.json();
}

export async function fetchCharts(ticker: string): Promise<ChartResponse> {
  const res = await fetch(
    `${API_URL}/api/screener/charts/${ticker.toUpperCase()}`,
  );
  if (!res.ok) throw new Error(`Failed to fetch charts: ${res.status}`);
  return res.json();
}

export async function fetchSectors(): Promise<{ sectors: SectorData[] }> {
  const res = await fetch(`${API_URL}/api/screener/sectors`);
  if (!res.ok) throw new Error(`Failed to fetch sectors: ${res.status}`);
  return res.json();
}

export async function compareStocks(
  tickers: string[],
): Promise<{ stocks: Stock[] }> {
  const tickerParam = tickers.join(",");
  const res = await fetch(
    `${API_URL}/api/screener/compare?tickers=${tickerParam}`,
  );
  if (!res.ok) throw new Error(`Failed to compare stocks: ${res.status}`);
  return res.json();
}

export async function healthCheck(): Promise<{ status: string }> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error("Backend not available");
  return res.json();
}

// ============================================================
// Personal Risk Capacity API (Phase 2)
// ============================================================

import type {
  RiskQuestionnaireRequest,
  RiskProfileResponse,
  PositionSizingRequest,
  PositionSizingResponse,
  SuitableStocksResponse,
  RiskMethodologyResponse,
} from "@/types/risk-profile";

export async function assessRiskProfile(
  data: RiskQuestionnaireRequest,
): Promise<RiskProfileResponse> {
  const res = await fetch(`${API_URL}/api/risk-profile/assess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Failed to assess risk profile: ${res.status}`);
  return res.json();
}

export async function getRiskProfile(
  profileId: string,
): Promise<RiskProfileResponse> {
  const res = await fetch(`${API_URL}/api/risk-profile/profile/${profileId}`);
  if (!res.ok) throw new Error(`Failed to get risk profile: ${res.status}`);
  return res.json();
}

export async function getPositionSizing(
  data: PositionSizingRequest,
): Promise<PositionSizingResponse> {
  const res = await fetch(`${API_URL}/api/risk-profile/position-sizing`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(`Failed to get position sizing: ${res.status}`);
  return res.json();
}

export async function getSuitableStocks(
  profileId: string,
  n: number = 10,
): Promise<SuitableStocksResponse> {
  const res = await fetch(
    `${API_URL}/api/risk-profile/suitable-stocks?profile_id=${profileId}&n=${n}`,
  );
  if (!res.ok) throw new Error(`Failed to get suitable stocks: ${res.status}`);
  return res.json();
}

export async function getRiskMethodology(): Promise<RiskMethodologyResponse> {
  const res = await fetch(`${API_URL}/api/risk-profile/methodology`);
  if (!res.ok) throw new Error(`Failed to get methodology: ${res.status}`);
  return res.json();
}

// ============================================================
// Pipeline API (Phase 3) - Quality Screening Pipeline
// ============================================================

export interface PillarScores {
  value: {
    score: number;
    weight: number;
    weighted_score: number;
    components: Record<
      string,
      { value: number; score: number; weight: number }
    >;
  };
  quality: {
    score: number;
    weight: number;
    weighted_score: number;
    components: Record<
      string,
      { value: number; score: number; weight: number }
    >;
  };
  growth: {
    score: number;
    weight: number;
    weighted_score: number;
    components: Record<
      string,
      { value: number; score: number; weight: number }
    >;
  };
  safety: {
    score: number;
    weight: number;
    weighted_score: number;
    components: Record<
      string,
      { value: number; score: number; weight: number }
    >;
  };
}

export interface Momentum {
  price_vs_200ma: number | null;
  price_vs_50ma: number | null;
  below_200ma: boolean | null;
  above_50ma: boolean | null;
  accumulation_zone: boolean | null;
  stabilizing: boolean | null;
  ideal_entry: boolean | null;
}

export interface AttentionStock {
  ticker: string;
  company_name: string;
  sector: string;
  triggers: Array<{
    type: string;
    triggered_at: string;
    metrics?: Record<string, number>;
    path?: number;
  }>;
  first_triggered: string;
  status: "active" | "graduated" | "expired";
}

export interface QualifiedStock {
  ticker: string;
  company_name: string;
  sector: string;
  industry: string;
  pillar_scores: PillarScores;
  composite_score: number;
  classification: "buy" | "hold" | "watch";
  current_price: number;
  fair_value: number;
  lstm_fair_value: number | null;
  margin_of_safety: number;
  lstm_predicted_growth: number | null;
  fcf_roic: number | null;
  revenue_growth: number | null;
  beta: number | null;
  momentum: Momentum;
  triggers: string[];
  analysis: {
    strengths: string[];
    weaknesses: string[];
    balanced: boolean;
  };
  last_updated: string;
}

export interface PipelineAttentionResponse {
  status: string;
  count: number;
  last_scan: string | null;
  stocks: AttentionStock[];
}

export interface PipelineQualifiedResponse {
  status: string;
  count: number;
  last_updated: string | null;
  stocks: QualifiedStock[];
}

export interface PipelineClassifiedResponse {
  status: string;
  counts: {
    buy?: number;
    hold?: number;
    watch?: number;
    total?: number;
  };
  classified: {
    buy: QualifiedStock[];
    hold: QualifiedStock[];
    watch: QualifiedStock[];
  };
}

export interface PipelineStockResponse {
  status: string;
  stock: QualifiedStock | null;
  in_attention: boolean;
  in_qualified: boolean;
}

export interface PipelineSummaryResponse {
  status: string;
  pipeline: {
    attention_active: number;
    attention_graduated: number;
    qualified_total: number;
    classifications: {
      buy?: number;
      hold?: number;
      watch?: number;
    };
    universe_size: number;
    last_scan?: {
      _id: string;
      scan_type: string;
      started_at: string;
      completed_at?: string;
      status: string;
    };
  };
}

// Pipeline API Functions

export async function fetchPipelineAttention(
  trigger?: string,
  status: string = "active",
): Promise<PipelineAttentionResponse> {
  const params = new URLSearchParams({ status });
  if (trigger) params.append("trigger", trigger);

  const res = await fetch(`${API_URL}/api/pipeline/attention?${params}`);
  if (!res.ok)
    throw new Error(`Failed to fetch attention stocks: ${res.status}`);
  return res.json();
}

export async function fetchPipelineQualified(
  classification?: "buy" | "hold" | "watch",
  sector?: string,
  minScore: number = 60,
): Promise<PipelineQualifiedResponse> {
  const params = new URLSearchParams({ min_score: minScore.toString() });
  if (classification) params.append("classification", classification);
  if (sector) params.append("sector", sector);

  const res = await fetch(`${API_URL}/api/pipeline/qualified?${params}`);
  if (!res.ok)
    throw new Error(`Failed to fetch qualified stocks: ${res.status}`);
  return res.json();
}

export async function fetchPipelineClassified(): Promise<PipelineClassifiedResponse> {
  const res = await fetch(`${API_URL}/api/pipeline/classified`);
  if (!res.ok)
    throw new Error(`Failed to fetch classified stocks: ${res.status}`);
  return res.json();
}

// Curated Watchlist Types
export interface CuratedStock {
  ticker: string;
  composite_score: number;
  margin_of_safety: number | null;
  fair_value?: number | null;
  current_price?: number | null;
  classification: "Buy" | "Hold" | "Watch";
  conviction: string; // "Strong", "Moderate", "Monitor", etc.
  pillars: Record<string, number>;
  sector: string;
  industry: string;
  market_cap: number | null;
  tiebreaker_score: number;
  excellent_pillars: number;
  justification: {
    short: string;
    long: string;
  };
  triggers: string[];
  watch_sub_category: string | null;
}

export interface CuratedWatchlistResponse {
  status: string;
  metadata: {
    generated_at: string;
    curation_version: string;
    source_qualified_count: number;
    curated_total: number;
    summary: {
      strong_buy: number;
      moderate_buy: number;
      hold: number;
      watch_by_category: Record<string, number>;
    };
  };
  strong_buy: CuratedStock[];
  moderate_buy: CuratedStock[];
  hold: CuratedStock[];
  watch: Record<string, CuratedStock[]>;
}

export async function fetchCuratedWatchlist(
  category?: string,
): Promise<CuratedWatchlistResponse> {
  const params = category ? `?category=${category}` : "";
  const res = await fetch(`${API_URL}/api/pipeline/curated${params}`);
  if (!res.ok)
    throw new Error(`Failed to fetch curated watchlist: ${res.status}`);
  return res.json();
}

export async function fetchPipelineStock(
  ticker: string,
): Promise<PipelineStockResponse> {
  const res = await fetch(
    `${API_URL}/api/pipeline/stock/${ticker.toUpperCase()}`,
  );
  if (!res.ok) throw new Error(`Failed to fetch pipeline stock: ${res.status}`);
  return res.json();
}

export async function fetchPipelineSummary(): Promise<PipelineSummaryResponse> {
  const res = await fetch(`${API_URL}/api/pipeline/summary`);
  if (!res.ok)
    throw new Error(`Failed to fetch pipeline summary: ${res.status}`);
  return res.json();
}

export async function triggerPipelineScan(
  scanType: "attention" | "qualified" = "attention",
  tickers?: string[],
): Promise<{ status: string; message: string }> {
  const params = new URLSearchParams({ scan_type: scanType });
  if (tickers) params.append("tickers", tickers.join(","));

  const res = await fetch(`${API_URL}/api/pipeline/trigger-scan?${params}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`Failed to trigger scan: ${res.status}`);
  return res.json();
}

// ============================================================
// AI Analysis API (Educational & Diagnostic)
// ============================================================

export interface PillarEducation {
  score: number;
  meaning: string;
  what_it_measures: string;
}

export interface EducationSection {
  available: boolean;
  message?: string;
  what_is_fair_value?: {
    title: string;
    content: string;
    key_insight: "undervalued" | "fairly_valued" | "overvalued";
  };
  pillar_scores_explained?: {
    title: string;
    pillars: Record<string, PillarEducation>;
  };
  classification_explained?: {
    title: string;
    current: string;
    meanings: Record<string, string>;
  };
  risk_education?: {
    title: string;
    beta_explained: string;
    diversification_tip: string;
  };
}

export interface SupportingMetric {
  name: string;
  value: string | number;
  note?: string;
}

export interface PillarStrengthWeakness {
  name: string;
  score: number;
  why_strong?: string;
  concern?: string;
  supporting_metrics?: SupportingMetric[];
}

export interface MetricsSnapshot {
  valuation: {
    current_price?: number;
    fair_value_estimate?: number;
    margin_of_safety_pct?: number;
    pe_ratio?: number;
    forward_pe?: number;
    pb_ratio?: number;
    peg_ratio?: number;
  };
  fundamentals: {
    roe_pct?: number;
    roa_pct?: number;
    profit_margin_pct?: number;
    gross_margin_pct?: number;
    current_ratio?: number;
    debt_to_equity?: number;
  };
  growth: {
    revenue_growth_pct?: number;
    earnings_growth_pct?: number;
  };
  risk: {
    beta?: number;
    dividend_yield_pct?: number;
  };
}

export interface DiagnosisSection {
  available: boolean;
  message?: string;
  overall_assessment?: {
    composite_score: number;
    interpretation: string;
    margin_of_safety: number;
    valuation_status: string;
  };
  metrics_snapshot?: MetricsSnapshot;
  strengths?: {
    count: number;
    pillars: PillarStrengthWeakness[];
  };
  weaknesses?: {
    count: number;
    pillars: PillarStrengthWeakness[];
  };
  sector_context?: {
    sector: string;
    note: string;
  };
  key_drivers?: string[];
  red_flags?: string[];
  catalysts?: string[];
}

export interface InvestmentThesisSection {
  available: boolean;
  message?: string;
  company?: string;
  ticker?: string;
  recommendation?: string;
  conviction?: "High" | "Medium-High" | "Medium" | "Low";
  bull_case?: {
    summary: string;
    points: string[];
  };
  bear_case?: {
    summary: string;
    points: string[];
  };
  base_case?: {
    expected_return: string;
    timeframe: string;
    key_assumption: string;
  };
  action_items?: string[];
}

export interface LLMSummarySection {
  available: boolean;
  message?: string;
  generated_by?: string;
  analysis?: string;
  error?: string;
  fallback?: string;
  context?: {
    ticker: string;
    composite_score: number;
    classification: string;
    margin_of_safety: number;
  };
  data_context?: {
    fair_value?: number;
    current_price?: number;
    consensus_score?: number;
    margin_of_safety?: number;
    classification?: string;
    sector?: string;
  };
}

export interface AIAnalysisResponse {
  status: string;
  ticker: string;
  generated_at: string;
  analysis_type?: "llm" | "rule_based";
  sections: {
    llm_summary?: LLMSummarySection;
    education?: EducationSection;
    diagnosis?: DiagnosisSection;
    investment_thesis?: InvestmentThesisSection;
  };
}

export interface QuickAnalysisResponse {
  status: string;
  ticker: string;
  summary: {
    headline: string;
    composite_score?: number;
    best_pillar?: { name: string; score: number };
    classification?: string;
    one_liner: string;
  };
  source: "pipeline" | "live";
}

export async function fetchAIAnalysis(
  ticker: string,
  options?: {
    includeEducation?: boolean;
    includeDiagnosis?: boolean;
    includeThesis?: boolean;
    useLLM?: boolean;
  },
): Promise<AIAnalysisResponse> {
  const params = new URLSearchParams();
  if (options?.includeEducation !== undefined) {
    params.append("include_education", String(options.includeEducation));
  }
  if (options?.includeDiagnosis !== undefined) {
    params.append("include_diagnosis", String(options.includeDiagnosis));
  }
  if (options?.includeThesis !== undefined) {
    params.append("include_thesis", String(options.includeThesis));
  }
  if (options?.useLLM !== undefined) {
    params.append("use_llm", String(options.useLLM));
  }

  const queryString = params.toString();
  const url = `${API_URL}/api/analysis/stock/${ticker.toUpperCase()}${
    queryString ? `?${queryString}` : ""
  }`;

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch AI analysis: ${res.status}`);
  return res.json();
}

export async function fetchQuickAnalysis(
  ticker: string,
): Promise<QuickAnalysisResponse> {
  const res = await fetch(
    `${API_URL}/api/analysis/stock/${ticker.toUpperCase()}/quick`,
  );
  if (!res.ok) throw new Error(`Failed to fetch quick analysis: ${res.status}`);
  return res.json();
}

// =============================================================================
// MULTI-AGENT ANALYSIS API
// =============================================================================

export interface MultiAgentSentiment {
  available: boolean;
  agent?: string;
  analysis?: string;
  sentiment_score?: number;
  sentiment_label?: string;
  error?: string;
}

export interface MultiAgentFundamentals {
  available: boolean;
  agent?: string;
  analysis?: string;
  quality_score?: number;
  growth_score?: number;
  value_score?: number;
  error?: string;
}

export interface MultiAgentMLValuation {
  available: boolean;
  agent?: string;
  analysis?: string;
  extracted_metrics?: {
    fair_value?: number | null;
    consensus_score?: number | null;
    margin_of_safety?: number | null;
  };
  error?: string;
}

export interface MultiAgentSynthesis {
  available: boolean;
  synthesis?: string;
  agents_combined?: string[];
  deep_mode?: boolean;
  contrarian_signal?: boolean;
  scores_summary?: {
    sentiment: number;
    quality: number;
    growth?: number | null;
    value?: number | null;
    fair_value: number | null;
    margin_of_safety: number | null;
  };
  error?: string;
}

export interface MultiAgentAnalysisResponse {
  status: string;
  ticker: string;
  generated_at: string;
  analysis_type: "multi_agent" | "multi_agent_deep";
  agents_used: string[];
  sections: {
    sentiment?: MultiAgentSentiment;
    fundamentals?: MultiAgentFundamentals;
    ml_valuation?: MultiAgentMLValuation;
    synthesis?: MultiAgentSynthesis;
    orchestrated_analysis?: {
      available: boolean;
      content?: string;
      agent?: string;
      error?: string;
    };
  };
}

export async function fetchMultiAgentAnalysis(
  ticker: string,
  options?: {
    includeSentiment?: boolean;
    includeFundamentals?: boolean;
    includeMLValuation?: boolean;
    deepAnalysis?: boolean;
    deepSynthesis?: boolean;
  },
): Promise<MultiAgentAnalysisResponse> {
  const params = new URLSearchParams();
  if (options?.includeSentiment !== undefined) {
    params.append("include_sentiment", String(options.includeSentiment));
  }
  if (options?.includeFundamentals !== undefined) {
    params.append("include_fundamentals", String(options.includeFundamentals));
  }
  if (options?.includeMLValuation !== undefined) {
    params.append("include_ml_valuation", String(options.includeMLValuation));
  }
  if (options?.deepAnalysis !== undefined) {
    params.append("deep_analysis", String(options.deepAnalysis));
  }
  if (options?.deepSynthesis !== undefined) {
    params.append("deep_synthesis", String(options.deepSynthesis));
  }

  const queryString = params.toString();
  const url = `${API_URL}/api/multiagent/stock/${ticker.toUpperCase()}${
    queryString ? `?${queryString}` : ""
  }`;

  const res = await fetch(url);
  if (!res.ok)
    throw new Error(`Failed to fetch multi-agent analysis: ${res.status}`);
  return res.json();
}

export async function fetchSentimentAnalysis(
  ticker: string,
): Promise<{ status: string; ticker: string; sentiment: MultiAgentSentiment }> {
  const res = await fetch(
    `${API_URL}/api/multiagent/stock/${ticker.toUpperCase()}/sentiment`,
  );
  if (!res.ok)
    throw new Error(`Failed to fetch sentiment analysis: ${res.status}`);
  return res.json();
}

export async function fetchFundamentalsAnalysis(ticker: string): Promise<{
  status: string;
  ticker: string;
  fundamentals: MultiAgentFundamentals;
}> {
  const res = await fetch(
    `${API_URL}/api/multiagent/stock/${ticker.toUpperCase()}/fundamentals`,
  );
  if (!res.ok)
    throw new Error(`Failed to fetch fundamentals analysis: ${res.status}`);
  return res.json();
}

export async function fetchMLValuationAnalysis(ticker: string): Promise<{
  status: string;
  ticker: string;
  ml_valuation: MultiAgentMLValuation;
}> {
  const res = await fetch(
    `${API_URL}/api/multiagent/stock/${ticker.toUpperCase()}/ml-valuation`,
  );
  if (!res.ok)
    throw new Error(`Failed to fetch ML valuation analysis: ${res.status}`);
  return res.json();
}
