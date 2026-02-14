/**
 * Client-Side Risk Scoring Engine
 *
 * Calculates personal risk profile locally without API calls.
 * Based on the Personal Risk Capacity Framework from backend.
 *
 * All data is stored in localStorage - no server-side user data storage.
 */

import type {
  RiskQuestionnaireRequest,
  RiskProfileResponse,
  ExperienceLevel,
  InvestmentHorizon,
  PanicSellResponse,
  BetaRange,
} from "@/types/risk-profile";

// Emotional Buffer Factors by Experience Level
const EMOTIONAL_BUFFER_FACTORS: Record<ExperienceLevel, number> = {
  first_time: 2.0,
  beginner: 1.75,
  intermediate: 1.5,
  experienced: 1.25,
  professional: 1.0,
};

// Base Margin of Safety Threshold (20% is commonly used)
const BASE_MOS_THRESHOLD = 20.0;

// Panic Sell Weights for tolerance scoring
const PANIC_SELL_WEIGHTS: Record<PanicSellResponse, number> = {
  sell_immediately: 0.0,
  sell_partial: 0.25,
  hold_wait: 0.5,
  buy_more: 0.75,
  no_panic: 1.0,
};

// Investment Horizon Risk Multipliers
const HORIZON_RISK_MULTIPLIERS: Record<InvestmentHorizon, number> = {
  short: 0.5,
  medium: 0.75,
  long: 1.0,
  very_long: 1.25,
};

// Beta ranges by profile type
const BETA_RANGES: Record<string, BetaRange> = {
  conservative: { min: 0.0, max: 0.8 },
  moderate: { min: 0.5, max: 1.2 },
  aggressive: { min: 0.8, max: 2.0 },
};

/**
 * Generate a short unique ID
 */
function generateProfileId(): string {
  return `${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .substring(2, 8)}`;
}

/**
 * Calculate Risk Capacity Score (financial ability to absorb losses)
 */
function calculateRiskCapacity(
  emergencyFundMonths: number,
  monthlyInvestmentPercent: number,
  maxTolerableLossPercent: number
): {
  score: number;
  max_loss_affordable: number;
  max_position_pct: number;
  reasoning: string;
} {
  // Emergency fund score (0-40 points)
  const efScore = Math.min(40, emergencyFundMonths * 6.67);

  // Monthly investment score (0-30 points)
  const investScore = Math.min(30, monthlyInvestmentPercent * 1.5);

  // Max loss score (0-30 points)
  const lossScore = Math.min(30, maxTolerableLossPercent * 0.6);

  const totalScore = efScore + investScore + lossScore;

  // Calculate max position % based on capacity
  const maxPositionPct = Math.min(20, 5 + totalScore / 10);

  const reasoningParts: string[] = [];
  if (emergencyFundMonths >= 6) {
    reasoningParts.push(
      `${emergencyFundMonths}-month emergency fund provides good buffer`
    );
  } else {
    reasoningParts.push("Consider building emergency fund to 6+ months");
  }

  if (monthlyInvestmentPercent >= 15) {
    reasoningParts.push(
      `${monthlyInvestmentPercent}% savings rate shows strong capacity`
    );
  }

  return {
    score: Math.round(totalScore * 10) / 10,
    max_loss_affordable: maxTolerableLossPercent,
    max_position_pct: Math.round(maxPositionPct * 10) / 10,
    reasoning: reasoningParts.join("; ") || "Standard risk capacity",
  };
}

/**
 * Calculate Risk Tolerance Score (emotional ability to handle volatility)
 */
function calculateRiskTolerance(
  panicSellResponse: PanicSellResponse,
  volatilityComfort: number,
  investmentHorizon: InvestmentHorizon
): {
  score: number;
  volatility_tolerance: string;
  panic_risk: string;
  reasoning: string;
} {
  // Panic response score (0-40 points)
  const panicWeight = PANIC_SELL_WEIGHTS[panicSellResponse] ?? 0.5;
  const panicScore = panicWeight * 40;

  // Volatility comfort score (0-30 points)
  const volScore = ((volatilityComfort - 1) / 4) * 30;

  // Horizon adjustment (0-30 points)
  const horizonMultiplier = HORIZON_RISK_MULTIPLIERS[investmentHorizon] ?? 0.75;
  const horizonScore = horizonMultiplier * 30;

  const totalScore = panicScore + volScore + horizonScore;

  // Determine volatility tolerance label
  let volTolerance = "moderate";
  if (volatilityComfort >= 4) volTolerance = "high";
  else if (volatilityComfort <= 2) volTolerance = "low";

  // Determine panic risk
  let panicRisk = "moderate";
  if (panicWeight >= 0.75) panicRisk = "low";
  else if (panicWeight <= 0.25) panicRisk = "high";

  const reasoningParts: string[] = [];
  if (panicRisk === "low") {
    reasoningParts.push("Strong emotional discipline during downturns");
  } else if (panicRisk === "high") {
    reasoningParts.push("May need extra safety margin for emotional comfort");
  }

  if (investmentHorizon === "long" || investmentHorizon === "very_long") {
    reasoningParts.push("Longer horizon allows weathering volatility");
  }

  return {
    score: Math.round(totalScore * 10) / 10,
    volatility_tolerance: volTolerance,
    panic_risk: panicRisk,
    reasoning: reasoningParts.join("; ") || "Standard risk tolerance",
  };
}

/**
 * Calculate Emotional Buffer based on experience level
 */
function calculateEmotionalBuffer(experienceLevel: ExperienceLevel): {
  factor: number;
  base_mos_threshold: number;
  adjusted_mos_threshold: number;
  reasoning: string;
} {
  const factor = EMOTIONAL_BUFFER_FACTORS[experienceLevel] ?? 1.5;
  const adjustedMoS = BASE_MOS_THRESHOLD * factor;

  const experienceLabels: Record<ExperienceLevel, string> = {
    first_time: "First-time investor needs maximum safety buffer",
    beginner: "Beginner benefits from higher margin of safety",
    intermediate: "Intermediate experience allows moderate buffer",
    experienced: "Experience allows slightly reduced buffer",
    professional: "Professional experience allows standard margin",
  };

  return {
    factor,
    base_mos_threshold: BASE_MOS_THRESHOLD,
    adjusted_mos_threshold: adjustedMoS,
    reasoning:
      experienceLabels[experienceLevel] ?? "Standard emotional buffer applied",
  };
}

/**
 * Classify overall profile based on average score
 */
function classifyProfile(
  avgScore: number
): "conservative" | "moderate" | "aggressive" {
  if (avgScore < 40) return "conservative";
  if (avgScore >= 70) return "aggressive";
  return "moderate";
}

/**
 * Generate personalized recommendations
 */
function generateRecommendations(
  profileType: string,
  experienceLevel: ExperienceLevel,
  emergencyFundMonths: number,
  maxLossPercent: number
): string[] {
  const recommendations: string[] = [];

  // Profile-based recommendation
  switch (profileType) {
    case "conservative":
      recommendations.push(
        "Focus on low-volatility stocks with beta < 0.8 and strong dividend history"
      );
      recommendations.push(
        "Prioritize stocks with high margin of safety (30%+) for extra protection"
      );
      break;
    case "moderate":
      recommendations.push(
        "Consider a balanced mix of value and growth stocks with moderate beta (0.5-1.2)"
      );
      recommendations.push(
        "Look for stocks with 20-30% margin of safety for optimal entry points"
      );
      break;
    case "aggressive":
      recommendations.push(
        "Growth-oriented stocks with higher beta may fit your risk capacity"
      );
      recommendations.push("Can consider stocks with 15-25% margin of safety");
      break;
  }

  // Emergency fund recommendation
  if (emergencyFundMonths < 6) {
    recommendations.push(
      "Consider building emergency fund to 6+ months before increasing stock allocation"
    );
  }

  // Experience-based recommendation
  if (experienceLevel === "first_time" || experienceLevel === "beginner") {
    recommendations.push(
      "Start with smaller position sizes until you gain more market experience"
    );
  }

  // Loss tolerance warning
  if (maxLossPercent < 15) {
    recommendations.push(
      "Your low loss tolerance suggests focusing on defensive sectors like utilities and consumer staples"
    );
  }

  return recommendations;
}

/**
 * Main function: Calculate complete risk profile from questionnaire data
 *
 * This runs entirely on the client-side and stores results in localStorage.
 */
export function calculateRiskProfile(
  data: RiskQuestionnaireRequest
): RiskProfileResponse {
  const profileId = generateProfileId();

  // Calculate component scores
  const riskCapacity = calculateRiskCapacity(
    data.emergency_fund_months,
    data.monthly_investment_percent,
    data.max_tolerable_loss_percent
  );

  const riskTolerance = calculateRiskTolerance(
    data.panic_sell_response,
    data.volatility_comfort,
    data.investment_horizon
  );

  const emotionalBuffer = calculateEmotionalBuffer(data.experience_level);

  // Determine overall profile
  const avgScore = (riskCapacity.score + riskTolerance.score) / 2;
  const overallProfile = classifyProfile(avgScore);

  // Get suitable beta range
  const betaRange = BETA_RANGES[overallProfile];

  // Generate recommendations
  const recommendations = generateRecommendations(
    overallProfile,
    data.experience_level,
    data.emergency_fund_months,
    data.max_tolerable_loss_percent
  );

  return {
    profile_id: profileId,
    created_at: new Date().toISOString(),
    risk_capacity: riskCapacity,
    risk_tolerance: riskTolerance,
    emotional_buffer: emotionalBuffer,
    overall_risk_profile: overallProfile,
    suitable_beta_range: betaRange,
    recommendations,
  };
}
