/**
 * Risk Assessment Modal
 *
 * Popup modal for Personal Risk Capacity assessment.
 * Shows on first visit or when user wants to retake assessment.
 * Can be skipped with default profile settings.
 *
 * UX Improvements:
 * - Skip button in footer for easier navigation
 * - Retake confirmation prompt
 * - All scoring done client-side (no API calls)
 */
"use client";

import * as React from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  useLocalRiskProfile,
  useAssessRiskProfile,
  useRiskProfile,
} from "@/hooks/useRiskProfile";
import type {
  RiskQuestionnaireRequest,
  ExperienceLevel,
  InvestmentHorizon,
  PanicSellResponse,
} from "@/types/risk-profile";
import {
  Shield,
  SkipForward,
  RefreshCw,
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  Info,
  Loader2,
} from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";

interface RiskAssessmentModalProps {
  /** External control for open state */
  open?: boolean;
  /** Callback when modal requests close */
  onOpenChange?: (open: boolean) => void;
  /** Force show modal even if profile exists */
  forceShow?: boolean;
}

export function RiskAssessmentModal({
  open: externalOpen,
  onOpenChange,
  forceShow = false,
}: RiskAssessmentModalProps) {
  const { hasProfile, refresh } = useLocalRiskProfile();
  const [internalOpen, setInternalOpen] = React.useState(false);
  const [showResults, setShowResults] = React.useState(false);
  const [completedProfileId, setCompletedProfileId] = React.useState<
    string | null
  >(null);
  const [showRetakeConfirm, setShowRetakeConfirm] = React.useState(false);

  // Check if first visit (no profile in localStorage)
  React.useEffect(() => {
    // Only auto-show on first visit, not for forced retakes
    if (!forceShow && !hasProfile && typeof window !== "undefined") {
      const hasSeenModal = localStorage.getItem("emetix_modal_seen");
      if (!hasSeenModal) {
        setInternalOpen(true);
      }
    }
  }, [hasProfile, forceShow]);

  // Support external control
  const isOpen = externalOpen !== undefined ? externalOpen : internalOpen;
  const setIsOpen = (value: boolean) => {
    if (onOpenChange) {
      onOpenChange(value);
    } else {
      setInternalOpen(value);
    }
  };

  const handleComplete = (profileId: string) => {
    setCompletedProfileId(profileId);
    setShowResults(true);
    refresh(); // Refresh the profile hook
    // Mark modal as seen
    if (typeof window !== "undefined") {
      localStorage.setItem("emetix_modal_seen", "true");
    }
  };

  const handleSkip = () => {
    // Just close modal - user views without personalized profile
    // Mark as seen so modal doesn't reappear
    if (typeof window !== "undefined") {
      localStorage.setItem("emetix_modal_seen", "true");
    }
    setIsOpen(false);
  };

  const handleDone = () => {
    setShowResults(false);
    setCompletedProfileId(null);
    setIsOpen(false);
  };

  // Show retake confirmation instead of immediately resetting
  const handleRetakeRequest = () => {
    setShowRetakeConfirm(true);
  };

  const handleRetakeConfirmed = () => {
    setShowRetakeConfirm(false);
    setShowResults(false);
    setCompletedProfileId(null);
    setIsOpen(true);
  };

  const handleRetakeCancelled = () => {
    setShowRetakeConfirm(false);
  };

  return (
    <>
      {/* Retake Confirmation Dialog */}
      <AlertDialog open={showRetakeConfirm} onOpenChange={setShowRetakeConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Retake Risk Assessment?</AlertDialogTitle>
            <AlertDialogDescription>
              This will replace your current risk profile with a new one. Your
              previous answers will not be saved.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={handleRetakeCancelled}>
              Keep Current Profile
            </AlertDialogCancel>
            <AlertDialogAction onClick={handleRetakeConfirmed}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Yes, Retake
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Main Assessment Modal */}
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          {!showResults ? (
            <>
              <DialogHeader className="pb-0">
                <div className="flex items-center gap-2">
                  <Shield className="h-5 w-5 text-primary" />
                  <DialogTitle>Personalize Your Experience</DialogTitle>
                </div>
                <DialogDescription>
                  Take a quick assessment to get personalized stock
                  recommendations based on your risk tolerance. This takes about
                  2 minutes.
                </DialogDescription>
              </DialogHeader>

              {/* Questionnaire */}
              <div className="py-2">
                <RiskQuestionnaireInline
                  onComplete={handleComplete}
                  onSkip={handleSkip}
                />
              </div>
            </>
          ) : (
            <>
              <DialogHeader>
                <DialogTitle>Your Risk Profile</DialogTitle>
                <DialogDescription>
                  Based on your responses, here is your personalized profile
                </DialogDescription>
              </DialogHeader>

              {completedProfileId && (
                <ProfileResultsInline profileId={completedProfileId} />
              )}

              <div className="flex justify-between pt-4">
                <Button variant="outline" onClick={handleRetakeRequest}>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Retake
                </Button>
                <Button onClick={handleDone}>View Recommendations</Button>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

// =============================================================================
// Inline Questionnaire Component (for modal use)
// =============================================================================

const questionnaireSchema = z.object({
  experience_level: z.enum([
    "first_time",
    "beginner",
    "intermediate",
    "experienced",
    "professional",
  ]),
  investment_horizon: z.enum(["short", "medium", "long", "very_long"]),
  emergency_fund_months: z.number().min(0).max(36),
  monthly_investment_percent: z.number().min(0).max(100),
  max_tolerable_loss_percent: z.number().min(0).max(100),
  panic_sell_response: z.enum([
    "sell_immediately",
    "sell_partial",
    "hold_wait",
    "buy_more",
    "no_panic",
  ]),
  volatility_comfort: z.number().min(1).max(5),
  portfolio_value: z.number().min(0).optional(),
  monthly_income: z.number().min(0).optional(),
});

type FormData = z.infer<typeof questionnaireSchema>;

const EXPERIENCE_OPTIONS: {
  value: ExperienceLevel;
  label: string;
  description: string;
}[] = [
  {
    value: "first_time",
    label: "First-time Investor",
    description: "Never invested before",
  },
  {
    value: "beginner",
    label: "Beginner",
    description: "Less than 1 year experience",
  },
  {
    value: "intermediate",
    label: "Intermediate",
    description: "1-3 years experience",
  },
  {
    value: "experienced",
    label: "Experienced",
    description: "3-10 years experience",
  },
  {
    value: "professional",
    label: "Professional",
    description: "10+ years or finance professional",
  },
];

const HORIZON_OPTIONS: {
  value: InvestmentHorizon;
  label: string;
  description: string;
}[] = [
  { value: "short", label: "Short-term", description: "Less than 1 year" },
  { value: "medium", label: "Medium-term", description: "1-5 years" },
  { value: "long", label: "Long-term", description: "5-10 years" },
  { value: "very_long", label: "Very Long-term", description: "10+ years" },
];

const PANIC_OPTIONS: {
  value: PanicSellResponse;
  label: string;
  description: string;
}[] = [
  {
    value: "sell_immediately",
    label: "Sell Everything",
    description: "I'd panic and sell immediately",
  },
  {
    value: "sell_partial",
    label: "Sell Partially",
    description: "I'd sell some to reduce exposure",
  },
  {
    value: "hold_wait",
    label: "Hold & Wait",
    description: "I'd hold and wait for recovery",
  },
  {
    value: "buy_more",
    label: "Buy More",
    description: "I'd see it as a buying opportunity",
  },
  {
    value: "no_panic",
    label: "No Concern",
    description: "I wouldn't be concerned at all",
  },
];

function RiskQuestionnaireInline({
  onComplete,
  onSkip,
}: {
  onComplete: (profileId: string) => void;
  onSkip: () => void;
}) {
  const [step, setStep] = React.useState(0);
  const { mutate, isPending } = useAssessRiskProfile();

  const form = useForm<FormData>({
    resolver: zodResolver(questionnaireSchema),
    defaultValues: {
      experience_level: "beginner",
      investment_horizon: "medium",
      emergency_fund_months: 3,
      monthly_investment_percent: 10,
      max_tolerable_loss_percent: 20,
      panic_sell_response: "hold_wait",
      volatility_comfort: 3,
      portfolio_value: 10000,
      monthly_income: 5000,
    },
  });

  const { watch, setValue, getValues } = form;
  const experienceLevel = watch("experience_level");
  const investmentHorizon = watch("investment_horizon");
  const emergencyFundMonths = watch("emergency_fund_months");
  const maxTolerableLossPercent = watch("max_tolerable_loss_percent");
  const panicSellResponse = watch("panic_sell_response");

  const steps = [
    { title: "Experience", field: "experience_level" },
    { title: "Time Horizon", field: "investment_horizon" },
    { title: "Financial Safety", field: "emergency_fund_months" },
    { title: "Loss Tolerance", field: "max_tolerable_loss_percent" },
    { title: "Market Reactions", field: "panic_sell_response" },
  ];

  const progress = ((step + 1) / steps.length) * 100;

  const handleNext = () => {
    if (step < steps.length - 1) {
      setStep(step + 1);
    } else {
      handleSubmit();
    }
  };

  const handleBack = () => {
    if (step > 0) setStep(step - 1);
  };

  const handleSubmit = () => {
    const formValues = getValues();
    const data: RiskQuestionnaireRequest = {
      ...formValues,
      portfolio_value: formValues.portfolio_value || 10000,
      monthly_income: formValues.monthly_income || 5000,
    };

    mutate(data, {
      onSuccess: (result) => {
        onComplete(result.profile_id);
      },
    });
  };

  const renderStep = () => {
    switch (step) {
      case 0:
        return (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-muted-foreground mb-3">
              <Info className="h-4 w-4" />
              <span className="text-sm">
                What is your investing experience?
              </span>
            </div>
            <RadioGroup
              value={experienceLevel}
              onValueChange={(v) =>
                setValue("experience_level", v as ExperienceLevel)
              }
              className="space-y-2"
            >
              {EXPERIENCE_OPTIONS.map((opt) => (
                <div
                  key={opt.value}
                  className="flex items-center space-x-3 rounded-lg border p-3 cursor-pointer hover:bg-muted/50 transition-colors"
                >
                  <RadioGroupItem value={opt.value} id={`exp-${opt.value}`} />
                  <Label
                    htmlFor={`exp-${opt.value}`}
                    className="flex-1 cursor-pointer"
                  >
                    <div className="font-medium text-sm">{opt.label}</div>
                    <div className="text-xs text-muted-foreground">
                      {opt.description}
                    </div>
                  </Label>
                </div>
              ))}
            </RadioGroup>
          </div>
        );

      case 1:
        return (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-muted-foreground mb-3">
              <Info className="h-4 w-4" />
              <span className="text-sm">
                How long do you plan to hold investments?
              </span>
            </div>
            <RadioGroup
              value={investmentHorizon}
              onValueChange={(v) =>
                setValue("investment_horizon", v as InvestmentHorizon)
              }
              className="space-y-2"
            >
              {HORIZON_OPTIONS.map((opt) => (
                <div
                  key={opt.value}
                  className="flex items-center space-x-3 rounded-lg border p-3 cursor-pointer hover:bg-muted/50 transition-colors"
                >
                  <RadioGroupItem value={opt.value} id={`hz-${opt.value}`} />
                  <Label
                    htmlFor={`hz-${opt.value}`}
                    className="flex-1 cursor-pointer"
                  >
                    <div className="font-medium text-sm">{opt.label}</div>
                    <div className="text-xs text-muted-foreground">
                      {opt.description}
                    </div>
                  </Label>
                </div>
              ))}
            </RadioGroup>
          </div>
        );

      case 2:
        return (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-3">
              <Info className="h-4 w-4" />
              <span className="text-sm">Emergency fund coverage</span>
            </div>
            <div className="space-y-6">
              <div>
                <Label className="text-sm font-medium">
                  Months of expenses saved: {emergencyFundMonths}
                </Label>
                <Slider
                  value={[emergencyFundMonths]}
                  onValueChange={([val]) =>
                    setValue("emergency_fund_months", val)
                  }
                  min={0}
                  max={12}
                  step={1}
                  className="mt-4"
                />
                <div className="flex justify-between text-xs text-muted-foreground mt-1">
                  <span>0 months</span>
                  <span>12+ months</span>
                </div>
              </div>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-3">
              <Info className="h-4 w-4" />
              <span className="text-sm">
                Max loss you can tolerate before selling
              </span>
            </div>
            <div>
              <Label className="text-sm font-medium">
                Maximum tolerable loss: {maxTolerableLossPercent}%
              </Label>
              <Slider
                value={[maxTolerableLossPercent]}
                onValueChange={([val]) =>
                  setValue("max_tolerable_loss_percent", val)
                }
                min={5}
                max={50}
                step={5}
                className="mt-4"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>5% (Very Conservative)</span>
                <span>50% (Very Aggressive)</span>
              </div>
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-muted-foreground mb-3">
              <Info className="h-4 w-4" />
              <span className="text-sm">
                If the market dropped 20%, what would you do?
              </span>
            </div>
            <RadioGroup
              value={panicSellResponse}
              onValueChange={(v) =>
                setValue("panic_sell_response", v as PanicSellResponse)
              }
              className="space-y-2"
            >
              {PANIC_OPTIONS.map((opt) => (
                <div
                  key={opt.value}
                  className="flex items-center space-x-3 rounded-lg border p-3 cursor-pointer hover:bg-muted/50 transition-colors"
                >
                  <RadioGroupItem value={opt.value} id={`panic-${opt.value}`} />
                  <Label
                    htmlFor={`panic-${opt.value}`}
                    className="flex-1 cursor-pointer"
                  >
                    <div className="font-medium text-sm">{opt.label}</div>
                    <div className="text-xs text-muted-foreground">
                      {opt.description}
                    </div>
                  </Label>
                </div>
              ))}
            </RadioGroup>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-3">
        <span className="text-sm font-medium">{steps[step].title}</span>
        <span className="text-xs text-muted-foreground">
          Step {step + 1} of {steps.length}
        </span>
      </div>
      <Progress value={progress} className="h-1.5 mb-4" />

      <div className="min-h-70">{renderStep()}</div>

      <Separator className="my-4" />

      {/* Footer with Skip button prominently placed */}
      <div className="flex justify-between items-center">
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleBack}
            disabled={step === 0}
          >
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onSkip}
            className="text-muted-foreground hover:text-foreground"
          >
            <SkipForward className="h-4 w-4 mr-1" />
            Skip for now
          </Button>
        </div>
        <Button size="sm" onClick={handleNext} disabled={isPending}>
          {isPending ? (
            <>
              <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              Analyzing...
            </>
          ) : step === steps.length - 1 ? (
            <>
              <CheckCircle className="h-4 w-4 mr-1" />
              Complete
            </>
          ) : (
            <>
              Next
              <ArrowRight className="h-4 w-4 ml-1" />
            </>
          )}
        </Button>
      </div>
    </div>
  );
}

// =============================================================================
// Inline Profile Results Component (for modal use)
// =============================================================================

function ProfileResultsInline({ profileId }: { profileId: string }) {
  const { data: profile, isLoading, error } = useRiskProfile(profileId);

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-6 w-32" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-3/4" />
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="text-sm text-muted-foreground">
        Unable to load profile
      </div>
    );
  }

  const getProfileColor = (type: string) => {
    switch (type?.toLowerCase()) {
      case "conservative":
        return "bg-blue-500";
      case "moderate":
        return "bg-yellow-500";
      case "aggressive":
        return "bg-orange-500";
      default:
        return "bg-gray-500";
    }
  };

  const profileType = profile.overall_risk_profile || "moderate";
  const betaRange = profile.suitable_beta_range || { min: 0.6, max: 1.3 };
  const adjustedMoS = profile.emotional_buffer?.adjusted_mos_threshold || 20;
  const maxPositionPct = profile.risk_capacity?.max_position_pct || 5;
  const recommendation =
    profile.recommendations?.[0] ||
    "Personalized recommendations based on your profile.";

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Badge className={getProfileColor(profileType)}>
          {profileType.replace("_", " ").toUpperCase()}
        </Badge>
        <span className="text-sm text-muted-foreground">Risk Profile</span>
      </div>

      <p className="text-sm">{recommendation}</p>

      <div className="grid grid-cols-2 gap-4 pt-2">
        <div className="bg-muted/50 rounded-lg p-3">
          <div className="text-xs text-muted-foreground">Beta Range</div>
          <div className="font-medium">
            {betaRange.min.toFixed(1)} - {betaRange.max.toFixed(1)}
          </div>
        </div>
        <div className="bg-muted/50 rounded-lg p-3">
          <div className="text-xs text-muted-foreground">
            Required Margin of Safety
          </div>
          <div className="font-medium">{adjustedMoS.toFixed(0)}%+</div>
        </div>
        <div className="bg-muted/50 rounded-lg p-3">
          <div className="text-xs text-muted-foreground">Max Position Size</div>
          <div className="font-medium">{maxPositionPct.toFixed(0)}%</div>
        </div>
        <div className="bg-muted/50 rounded-lg p-3">
          <div className="text-xs text-muted-foreground">Stocks That Match</div>
          <div className="font-medium text-green-600">View on Dashboard</div>
        </div>
      </div>
    </div>
  );
}
