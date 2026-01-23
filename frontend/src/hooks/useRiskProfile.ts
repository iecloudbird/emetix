/**
 * Risk Profile Hooks
 *
 * React hooks for Personal Risk Capacity Framework
 *
 * Profile data is stored in localStorage (no server-side storage):
 * - emetix_profile_id: Unique profile ID
 * - emetix_risk_profile: Full profile data (JSON)
 *
 * All risk scoring is done client-side for privacy.
 */

import { useMutation, useQuery } from "@tanstack/react-query";
import { useState, useEffect, useCallback } from "react";
import { calculateRiskProfile } from "@/lib/risk-scoring";
import type {
  RiskQuestionnaireRequest,
  RiskProfileResponse,
} from "@/types/risk-profile";

// localStorage keys
const PROFILE_ID_KEY = "emetix_profile_id";
const PROFILE_DATA_KEY = "emetix_risk_profile";

/**
 * Get stored profile data from localStorage
 */
function getStoredProfile(): RiskProfileResponse | null {
  if (typeof window === "undefined") return null;
  try {
    const data = localStorage.getItem(PROFILE_DATA_KEY);
    return data ? JSON.parse(data) : null;
  } catch {
    return null;
  }
}

/**
 * Save profile data to localStorage
 */
export function saveProfile(profile: RiskProfileResponse): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(PROFILE_ID_KEY, profile.profile_id);
  localStorage.setItem(PROFILE_DATA_KEY, JSON.stringify(profile));
}

/**
 * Clear stored profile
 */
export function clearStoredProfile(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(PROFILE_ID_KEY);
  localStorage.removeItem(PROFILE_DATA_KEY);
}

/**
 * Hook to submit risk questionnaire and get profile
 * Calculates entirely client-side - no API calls
 */
export function useAssessRiskProfile() {
  return useMutation({
    mutationFn: async (
      data: RiskQuestionnaireRequest
    ): Promise<RiskProfileResponse> => {
      // Calculate profile locally - no API call needed
      const profile = calculateRiskProfile(data);
      // Save to localStorage
      saveProfile(profile);
      return profile;
    },
  });
}

/**
 * Hook to get existing risk profile from localStorage
 * No API calls - all data is local
 */
export function useRiskProfile(profileId: string | null) {
  const [localProfile, setLocalProfile] = useState<RiskProfileResponse | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const stored = getStoredProfile();
    if (stored && (!profileId || stored.profile_id === profileId)) {
      setLocalProfile(stored);
    }
    setIsLoading(false);
  }, [profileId]);

  return {
    data: localProfile,
    isLoading,
    error: null,
  };
}

/**
 * Hook to get the locally stored profile without API call
 * Use this when you just need the cached profile data
 */
export function useLocalRiskProfile() {
  const [profile, setProfile] = useState<RiskProfileResponse | null>(null);
  const [profileId, setProfileId] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  // Refresh function to reload from localStorage
  const refresh = useCallback(() => {
    setRefreshKey((k) => k + 1);
  }, []);

  useEffect(() => {
    const stored = getStoredProfile();
    setProfile(stored);
    setProfileId(stored?.profile_id || null);
  }, [refreshKey]);

  // Normalize the profile to a common format for components
  const normalizedProfile = profile
    ? {
        profileId: profile.profile_id,
        profileType: profile.overall_risk_profile || "moderate",
        betaRange: profile.suitable_beta_range,
        requiredMoS: profile.emotional_buffer?.adjusted_mos_threshold ?? 20,
        maxPositionPercent: profile.risk_capacity?.max_position_pct ?? 5,
        portfolioValue: 50000, // Default, could be stored separately
        recommendation:
          profile.recommendations?.[0] ||
          "Personalized recommendations based on your profile.",
      }
    : null;

  // Clear and refresh profile
  const clearProfile = useCallback(() => {
    clearStoredProfile();
    setProfile(null);
    setProfileId(null);
  }, []);

  return {
    profile: normalizedProfile,
    profileId,
    hasProfile: !!profile,
    clearProfile,
    refresh,
    rawProfile: profile, // Original API response
  };
}

// Position sizing result type
interface PositionSizingResult {
  ticker: string;
  max_position_percent: number;
  max_position_value: number;
  max_shares: number;
  risk_factors: string[];
  recommendation: string;
  methodology: string;
}

/**
 * Hook to calculate position sizing for a stock
 * Uses local profile data to determine appropriate position size
 * Returns a mutation-like interface for compatibility
 */
export function usePositionSizing() {
  const { rawProfile } = useLocalRiskProfile();
  const [data, setData] = useState<PositionSizingResult | null>(null);
  const [isPending, setIsPending] = useState(false);

  const mutate = useCallback(
    (params: {
      profile_id?: string;
      ticker?: string;
      current_price: number;
      beta: number;
      margin_of_safety: number;
    }) => {
      if (!rawProfile) {
        setData(null);
        return;
      }

      setIsPending(true);

      const maxPositionPct = rawProfile.risk_capacity?.max_position_pct ?? 5;
      const betaRange = rawProfile.suitable_beta_range ?? {
        min: 0.5,
        max: 1.2,
      };
      const portfolioValue = 50000; // Default portfolio value

      // Adjust position based on beta
      let betaAdjustment = 1.0;
      if (params.beta < betaRange.min) {
        betaAdjustment = 1.1; // Slightly larger for low beta
      } else if (params.beta > betaRange.max) {
        betaAdjustment = 0.7; // Reduce for high beta
      }

      // Adjust for margin of safety
      const mosThreshold =
        rawProfile.emotional_buffer?.adjusted_mos_threshold ?? 20;
      const mosAdjustment = params.margin_of_safety >= mosThreshold ? 1.0 : 0.8;

      const adjustedPositionPct =
        maxPositionPct * betaAdjustment * mosAdjustment;
      const maxPositionValue = portfolioValue * (adjustedPositionPct / 100);
      const maxShares = Math.floor(maxPositionValue / params.current_price);

      // Build recommendation string
      let recommendation = "";
      if (adjustedPositionPct >= maxPositionPct) {
        recommendation = "This stock fits well within your risk parameters.";
      } else {
        recommendation = "Consider a smaller position due to elevated risk.";
      }

      const result: PositionSizingResult = {
        ticker: params.ticker || "",
        max_position_percent: adjustedPositionPct,
        max_position_value: maxPositionValue,
        max_shares: maxShares,
        risk_factors:
          params.beta > betaRange.max
            ? ["High beta relative to your profile"]
            : [],
        recommendation,
        methodology: "client-side calculation based on local risk profile",
      };

      setData(result);
      setIsPending(false);
    },
    [rawProfile]
  );

  return {
    mutate,
    data,
    isPending,
    reset: () => setData(null),
  };
}

/**
 * Helper hook to get stored profile ID
 * @deprecated Use useLocalRiskProfile() instead for full profile data
 */
export function useStoredProfileId(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(PROFILE_ID_KEY);
}

/**
 * Check if user has a stored risk profile
 */
export function hasStoredProfile(): boolean {
  if (typeof window === "undefined") return false;
  return localStorage.getItem(PROFILE_DATA_KEY) !== null;
}
