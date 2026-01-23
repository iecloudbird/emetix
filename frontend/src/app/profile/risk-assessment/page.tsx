/**
 * Risk Assessment Page
 *
 * Personal Risk Capacity questionnaire and results display.
 * Phase 2 core feature for thesis.
 *
 * UX States:
 * 1. No profile: Show invitation to take assessment
 * 2. Has profile: Show profile results with recommended stocks section
 * 3. Retake: Confirm before resetting
 */
"use client";

import * as React from "react";
import { RiskQuestionnaire } from "@/components/risk-profile/RiskQuestionnaire";
import { ProfileResults } from "@/components/risk-profile/ProfileResults";
import { useLocalRiskProfile } from "@/hooks/useRiskProfile";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
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
import {
  Shield,
  RefreshCw,
  ArrowRight,
  CheckCircle2,
  ClipboardList,
  Target,
  TrendingUp,
} from "lucide-react";
import Link from "next/link";

export default function RiskAssessmentPage() {
  const { hasProfile, rawProfile, clearProfile, refresh, profile } =
    useLocalRiskProfile();
  const [showQuestionnaire, setShowQuestionnaire] = React.useState(false);
  const [showRetakeConfirm, setShowRetakeConfirm] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(true);

  // Check for existing profile on mount
  React.useEffect(() => {
    setIsLoading(false);
  }, []);

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const handleComplete = (_profileId: string) => {
    setShowQuestionnaire(false);
    refresh(); // Refresh the profile hook to get updated data
  };

  const handleStartAssessment = () => {
    setShowQuestionnaire(true);
  };

  const handleRetakeRequest = () => {
    setShowRetakeConfirm(true);
  };

  const handleRetakeConfirmed = () => {
    clearProfile();
    setShowRetakeConfirm(false);
    setShowQuestionnaire(true);
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="py-8">
        <div className="max-w-2xl mx-auto space-y-6">
          <Card>
            <CardHeader>
              <Skeleton className="h-8 w-48" />
              <Skeleton className="h-4 w-32 mt-2" />
            </CardHeader>
            <CardContent className="space-y-4">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-2 w-full" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-2 w-full" />
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  // Taking questionnaire
  if (showQuestionnaire) {
    return (
      <div className="py-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">Personal Risk Assessment</h1>
          <p className="text-muted-foreground max-w-lg mx-auto">
            Answer 5 quick questions to get personalized stock recommendations
            matched to your risk capacity and investment experience.
          </p>
        </div>
        <RiskQuestionnaire onComplete={handleComplete} />

        {/* Educational Note */}
        <div className="max-w-2xl mx-auto mt-8 p-4 rounded-lg border bg-muted/50">
          <h3 className="font-semibold mb-2">Why This Matters</h3>
          <p className="text-sm text-muted-foreground">
            The same stock can be safe for one investor and risky for another.
            This assessment matches stock recommendations to{" "}
            <strong>your</strong> specific situation—considering your
            experience, financial cushion, and emotional comfort with
            volatility.
          </p>
        </div>
      </div>
    );
  }

  // Has profile - show results
  if (hasProfile && rawProfile) {
    return (
      <>
        {/* Retake Confirmation Dialog */}
        <AlertDialog
          open={showRetakeConfirm}
          onOpenChange={setShowRetakeConfirm}
        >
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Retake Risk Assessment?</AlertDialogTitle>
              <AlertDialogDescription>
                This will replace your current risk profile with a new one. Your
                previous answers will not be saved.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Keep Current Profile</AlertDialogCancel>
              <AlertDialogAction onClick={handleRetakeConfirmed}>
                <RefreshCw className="h-4 w-4 mr-2" />
                Yes, Retake
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <div className="py-8">
          {/* Header with completion status */}
          <div className="max-w-2xl mx-auto mb-8">
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                  <span className="text-sm text-green-600 font-medium">
                    Assessment Complete
                  </span>
                </div>
                <h1 className="text-3xl font-bold mb-2">Your Risk Profile</h1>
                <p className="text-muted-foreground">
                  Personalized investment recommendations based on your capacity
                </p>
              </div>
              <Button variant="outline" onClick={handleRetakeRequest}>
                <RefreshCw className="h-4 w-4 mr-2" />
                Retake
              </Button>
            </div>
          </div>

          {/* Profile Results */}
          <ProfileResults profile={rawProfile} />

          {/* Recommended Stocks Section */}
          <div className="max-w-2xl mx-auto mt-8">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-primary" />
                  Stocks Matching Your Profile
                </CardTitle>
                <CardDescription>
                  Based on your{" "}
                  <Badge variant="outline" className="mx-1">
                    {profile?.profileType}
                  </Badge>{" "}
                  risk profile, we recommend stocks with beta{" "}
                  {profile?.betaRange?.min.toFixed(1)} -{" "}
                  {profile?.betaRange?.max.toFixed(1)} and margin of safety
                  above {profile?.requiredMoS.toFixed(0)}%
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <p className="text-sm text-muted-foreground">
                    View the watchlist filtered to show only stocks that match
                    your risk capacity and investment style.
                  </p>
                  <div className="flex gap-4">
                    <Link href="/" className="flex-1">
                      <Button className="w-full">
                        <TrendingUp className="h-4 w-4 mr-2" />
                        View Suitable Stocks
                        <ArrowRight className="h-4 w-4 ml-2" />
                      </Button>
                    </Link>
                    <Link href="/pipeline">
                      <Button variant="outline">View Pipeline</Button>
                    </Link>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </>
    );
  }

  // No profile - invitation to take assessment
  return (
    <div className="py-8">
      <div className="max-w-2xl mx-auto">
        {/* Hero Section */}
        <Card className="mb-8 border-2 border-primary/20 bg-primary/5">
          <CardHeader className="text-center pb-2">
            <div className="mx-auto w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
              <Shield className="h-6 w-6 text-primary" />
            </div>
            <CardTitle className="text-2xl">
              Personalize Your Stock Recommendations
            </CardTitle>
            <CardDescription className="text-base">
              Take a 2-minute assessment to get stock picks matched to your risk
              capacity
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center">
            <Button size="lg" onClick={handleStartAssessment}>
              <ClipboardList className="h-4 w-4 mr-2" />
              Start Assessment
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </CardContent>
        </Card>

        {/* Benefits */}
        <div className="grid gap-4 sm:grid-cols-3 mb-8">
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <div className="mx-auto w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center mb-3">
                  <Target className="h-5 w-5 text-blue-600" />
                </div>
                <h3 className="font-semibold mb-1">Personalized Picks</h3>
                <p className="text-sm text-muted-foreground">
                  Stocks filtered to match your risk tolerance
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <div className="mx-auto w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center mb-3">
                  <Shield className="h-5 w-5 text-green-600" />
                </div>
                <h3 className="font-semibold mb-1">Emotional Buffer</h3>
                <p className="text-sm text-muted-foreground">
                  Higher safety margins for less experienced investors
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <div className="mx-auto w-10 h-10 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center mb-3">
                  <TrendingUp className="h-5 w-5 text-purple-600" />
                </div>
                <h3 className="font-semibold mb-1">Position Sizing</h3>
                <p className="text-sm text-muted-foreground">
                  Know how much to invest in each stock
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Without Profile - Show generic recommendations */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Without a Profile</CardTitle>
            <CardDescription>
              You can still browse all stocks, but you will not see personalized
              recommendations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-4">
              <Link href="/" className="flex-1">
                <Button variant="outline" className="w-full">
                  Browse All Stocks
                </Button>
              </Link>
              <Link href="/pipeline">
                <Button variant="outline">View Pipeline</Button>
              </Link>
            </div>
          </CardContent>
        </Card>

        {/* Educational Note */}
        <div className="mt-8 p-4 rounded-lg border bg-muted/50">
          <h3 className="font-semibold mb-2 flex items-center gap-2">
            <Shield className="h-4 w-4" />
            Why Risk Assessment Matters
          </h3>
          <p className="text-sm text-muted-foreground">
            The same stock can be safe for one investor and risky for another.
            Our assessment considers your experience level, financial cushion,
            and emotional comfort with volatility to recommend stocks that truly
            fit <strong>your</strong> situation—not just any undervalued stock.
          </p>
        </div>
      </div>
    </div>
  );
}
