/**
 * Risk Profile Types
 *
 * TypeScript interfaces for Personal Risk Capacity Framework
 */

// Experience levels for questionnaire
export type ExperienceLevel =
  | "first_time"
  | "beginner"
  | "intermediate"
  | "experienced"
  | "professional";

// Investment horizon options
export type InvestmentHorizon = "short" | "medium" | "long" | "very_long";

// Panic sell response options
export type PanicSellResponse =
  | "sell_immediately"
  | "sell_partial"
  | "hold_wait"
  | "buy_more"
  | "no_panic";

// Risk Questionnaire Request
export interface RiskQuestionnaireRequest {
  experience_level: ExperienceLevel;
  investment_horizon: InvestmentHorizon;
  emergency_fund_months: number;
  monthly_investment_percent: number;
  max_tolerable_loss_percent: number;
  panic_sell_response: PanicSellResponse;
  volatility_comfort: number; // 1-5
  portfolio_value?: number;
  monthly_income?: number;
}

// Beta Range for suitable stocks
export interface BetaRange {
  min: number;
  max: number;
}

// Risk Capacity Score (from API)
export interface RiskCapacityScore {
  score: number;
  max_loss_affordable?: number;
  max_position_pct?: number;
  reasoning: string;
}

// Risk Tolerance Score (from API)
export interface RiskToleranceScore {
  score: number;
  volatility_tolerance?: string;
  panic_risk?: string;
  reasoning: string;
}

// Emotional Buffer Result (from API)
export interface EmotionalBufferResult {
  factor: number;
  base_mos_threshold: number;
  adjusted_mos_threshold: number;
  reasoning: string;
}

// Risk Profile Response - matches actual API response
export interface RiskProfileResponse {
  profile_id: string;
  created_at?: string;
  risk_capacity: RiskCapacityScore;
  risk_tolerance: RiskToleranceScore;
  emotional_buffer: EmotionalBufferResult;
  overall_risk_profile: "conservative" | "moderate" | "aggressive";
  suitable_beta_range: BetaRange;
  recommendations: string[];
}

// Position Sizing Request
export interface PositionSizingRequest {
  profile_id: string;
  ticker: string;
  current_price: number;
  margin_of_safety: number;
  beta: number;
}

// Position Sizing Response
export interface PositionSizingResponse {
  ticker: string;
  max_position_percent: number;
  max_position_value: number;
  max_shares: number;
  risk_factors: string[];
  recommendation: string;
  methodology: string;
}

// Stock with suitability info
export interface SuitableStock {
  ticker: string;
  company_name: string;
  suitability: "excellent" | "good" | "moderate" | "poor";
  beta: number;
  margin_of_safety: number;
  valuation_score: number;
  recommendation: string;
  position_sizing?: {
    max_position_percent: number;
    max_shares: number;
  };
}

// Suitable Stocks Response
export interface SuitableStocksResponse {
  profile_id: string;
  suitable_count: number;
  total_screened: number;
  filters_applied: {
    beta_range: BetaRange;
    min_mos: number;
  };
  stocks: SuitableStock[];
}

// Methodology Response
export interface RiskMethodologyResponse {
  framework_name: string;
  version: string;
  components: {
    risk_capacity: string;
    risk_tolerance: string;
    emotional_buffer: string;
    position_sizing: string;
  };
  emotional_buffer_factors: Record<ExperienceLevel, number>;
  base_mos_threshold: number;
}
