/**
 * Risk Questionnaire Component
 *
 * 7-question form for Personal Risk Capacity assessment.
 * Progressive disclosure with clear explanations.
 */
"use client";

import * as React from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { useAssessRiskProfile } from "@/hooks/useRiskProfile";
import type {
  RiskQuestionnaireRequest,
  ExperienceLevel,
  InvestmentHorizon,
  PanicSellResponse,
} from "@/types/risk-profile";
import {
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  Info,
  Loader2,
} from "lucide-react";

// Form validation schema
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

interface Props {
  onComplete: (profileId: string) => void;
}

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

export function RiskQuestionnaire({ onComplete }: Props) {
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
  // Watch specific fields to trigger re-renders when they change
  const experienceLevel = watch("experience_level");
  const investmentHorizon = watch("investment_horizon");
  const emergencyFundMonths = watch("emergency_fund_months");
  const monthlyInvestmentPercent = watch("monthly_investment_percent");
  const maxTolerableLossPercent = watch("max_tolerable_loss_percent");
  const panicSellResponse = watch("panic_sell_response");
  const volatilityComfort = watch("volatility_comfort");

  const steps = [
    { title: "Experience", field: "experience_level" },
    { title: "Time Horizon", field: "investment_horizon" },
    { title: "Financial Safety", field: "emergency_fund_months" },
    { title: "Investment Capacity", field: "monthly_investment_percent" },
    { title: "Loss Tolerance", field: "max_tolerable_loss_percent" },
    { title: "Market Reactions", field: "panic_sell_response" },
    { title: "Volatility Comfort", field: "volatility_comfort" },
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
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-4">
              <Info className="h-4 w-4" />
              <span className="text-sm">
                This helps us adjust your safety margins
              </span>
            </div>
            <RadioGroup
              value={experienceLevel}
              onValueChange={(v) =>
                setValue("experience_level", v as ExperienceLevel)
              }
              className="space-y-3"
            >
              {EXPERIENCE_OPTIONS.map((opt) => (
                <div
                  key={opt.value}
                  className="flex items-center space-x-3 rounded-lg border p-4 cursor-pointer hover:bg-muted/50 transition-colors"
                >
                  <RadioGroupItem value={opt.value} id={opt.value} />
                  <Label htmlFor={opt.value} className="flex-1 cursor-pointer">
                    <div className="font-medium">{opt.label}</div>
                    <div className="text-sm text-muted-foreground">
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
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-4">
              <Info className="h-4 w-4" />
              <span className="text-sm">
                Longer horizons can tolerate more volatility
              </span>
            </div>
            <RadioGroup
              value={investmentHorizon}
              onValueChange={(v) =>
                setValue("investment_horizon", v as InvestmentHorizon)
              }
              className="space-y-3"
            >
              {HORIZON_OPTIONS.map((opt) => (
                <div
                  key={opt.value}
                  className="flex items-center space-x-3 rounded-lg border p-4 cursor-pointer hover:bg-muted/50 transition-colors"
                >
                  <RadioGroupItem value={opt.value} id={opt.value} />
                  <Label htmlFor={opt.value} className="flex-1 cursor-pointer">
                    <div className="font-medium">{opt.label}</div>
                    <div className="text-sm text-muted-foreground">
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
          <div className="space-y-6">
            <div className="flex items-center gap-2 text-muted-foreground mb-4">
              <Info className="h-4 w-4" />
              <span className="text-sm">
                Emergency fund protects you from forced selling
              </span>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between">
                <Label>Emergency Fund Coverage</Label>
                <span className="text-2xl font-bold text-primary">
                  {emergencyFundMonths} months
                </span>
              </div>
              <Slider
                value={[emergencyFundMonths]}
                onValueChange={([v]) => setValue("emergency_fund_months", v)}
                min={0}
                max={36}
                step={1}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>0 months</span>
                <span className="text-green-500">Recommended: 6+ months</span>
                <span>36 months</span>
              </div>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <div className="flex items-center gap-2 text-muted-foreground mb-4">
              <Info className="h-4 w-4" />
              <span className="text-sm">
                Higher capacity = more aggressive allocations
              </span>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between">
                <Label>Monthly Income You Can Invest</Label>
                <span className="text-2xl font-bold text-primary">
                  {monthlyInvestmentPercent}%
                </span>
              </div>
              <Slider
                value={[monthlyInvestmentPercent]}
                onValueChange={([v]) =>
                  setValue("monthly_investment_percent", v)
                }
                min={0}
                max={50}
                step={5}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>0%</span>
                <span className="text-green-500">Typical: 10-20%</span>
                <span>50%</span>
              </div>
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-6">
            <div className="flex items-center gap-2 text-muted-foreground mb-4">
              <Info className="h-4 w-4" />
              <span className="text-sm">
                This determines your risk tolerance score
              </span>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between">
                <Label>Maximum Portfolio Loss You Can Tolerate</Label>
                <span className="text-2xl font-bold text-primary">
                  {maxTolerableLossPercent}%
                </span>
              </div>
              <Slider
                value={[maxTolerableLossPercent]}
                onValueChange={([v]) =>
                  setValue("max_tolerable_loss_percent", v)
                }
                min={5}
                max={50}
                step={5}
                className="w-full"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span className="text-green-500">Conservative: 5-15%</span>
                <span className="text-yellow-500">Moderate: 15-30%</span>
                <span className="text-red-500">Aggressive: 30%+</span>
              </div>
            </div>
          </div>
        );

      case 5:
        return (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-4">
              <Info className="h-4 w-4" />
              <span className="text-sm">
                If your portfolio dropped 20% in one week...
              </span>
            </div>
            <RadioGroup
              value={panicSellResponse}
              onValueChange={(v) =>
                setValue("panic_sell_response", v as PanicSellResponse)
              }
              className="space-y-3"
            >
              {PANIC_OPTIONS.map((opt) => (
                <div
                  key={opt.value}
                  className="flex items-center space-x-3 rounded-lg border p-4 cursor-pointer hover:bg-muted/50 transition-colors"
                >
                  <RadioGroupItem value={opt.value} id={opt.value} />
                  <Label htmlFor={opt.value} className="flex-1 cursor-pointer">
                    <div className="font-medium">{opt.label}</div>
                    <div className="text-sm text-muted-foreground">
                      {opt.description}
                    </div>
                  </Label>
                </div>
              ))}
            </RadioGroup>
          </div>
        );

      case 6:
        return (
          <div className="space-y-6">
            <div className="flex items-center gap-2 text-muted-foreground mb-4">
              <Info className="h-4 w-4" />
              <span className="text-sm">
                How comfortable are you seeing daily fluctuations?
              </span>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <Label>Volatility Comfort Level</Label>
                <div className="flex gap-1">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      type="button"
                      onClick={() => setValue("volatility_comfort", star)}
                      className={`text-2xl transition-colors ${
                        star <= volatilityComfort
                          ? "text-yellow-500"
                          : "text-muted-foreground/30"
                      }`}
                    >
                      ★
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex justify-between text-sm text-muted-foreground pt-2">
                <span>Very Uncomfortable</span>
                <span>Very Comfortable</span>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <Card className="max-w-2xl mx-auto">
      <CardHeader>
        <div className="flex justify-between items-center mb-2">
          <CardTitle className="text-xl">Personal Risk Assessment</CardTitle>
          <span className="text-sm text-muted-foreground">
            Step {step + 1} of {steps.length}
          </span>
        </div>
        <Progress value={progress} className="h-2" />
        <CardDescription className="pt-2">{steps[step].title}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="min-h-75">{renderStep()}</div>

        <Separator className="my-6" />

        <div className="flex justify-between">
          <Button variant="outline" onClick={handleBack} disabled={step === 0}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <Button onClick={handleNext} disabled={isPending}>
            {isPending ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : step === steps.length - 1 ? (
              <>
                <CheckCircle className="h-4 w-4 mr-2" />
                Complete Assessment
              </>
            ) : (
              <>
                Next
                <ArrowRight className="h-4 w-4 ml-2" />
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
