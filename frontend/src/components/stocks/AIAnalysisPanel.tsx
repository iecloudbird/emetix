/**
 * AIAnalysisPanel - Comprehensive AI-driven stock analysis
 *
 * Displays three sections:
 * 1. Education - What the metrics mean (for beginners)
 * 2. Diagnosis - Why the stock is valued this way
 * 3. Investment Thesis - Bull case vs Bear case
 */
"use client";

import React, { useState, useEffect } from "react";
import { useAIAnalysis, getPillarColor } from "@/hooks/use-pipeline";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  GraduationCap,
  Stethoscope,
  Scale,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Lightbulb,
  Target,
  Shield,
  BarChart3,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type {
  EducationSection,
  DiagnosisSection,
  InvestmentThesisSection,
  LLMSummarySection,
} from "@/lib/api";
import { fetchAIAnalysis, type AIAnalysisResponse } from "@/lib/api";
import { useQuery } from "@tanstack/react-query";

interface AIAnalysisPanelProps {
  ticker: string;
  className?: string;
}

export function AIAnalysisPanel({ ticker, className }: AIAnalysisPanelProps) {
  const [activeTab, setActiveTab] = useState<
    "ai" | "thesis" | "diagnosis" | "education"
  >("thesis");
  const [llmEnabled, setLlmEnabled] = useState(false);

  // Rule-based analysis (fast)
  const { data, isLoading, error } = useAIAnalysis(ticker, {
    includeEducation: true,
    includeDiagnosis: true,
    includeThesis: true,
  });

  // LLM analysis (on-demand, slower)
  const {
    data: llmData,
    isLoading: llmLoading,
    refetch: fetchLlm,
  } = useQuery<AIAnalysisResponse>({
    queryKey: ["ai-analysis-llm", ticker],
    queryFn: () => fetchAIAnalysis(ticker, { useLLM: true }),
    enabled: false, // Only fetch when user clicks
    staleTime: 30 * 60 * 1000, // 30 minutes
  });

  const handleEnableLLM = () => {
    setLlmEnabled(true);
    setActiveTab("ai");
    fetchLlm();
  };

  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-72" />
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-32 w-full" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={cn("border-red-200", className)}>
        <CardContent className="py-6">
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Analysis Unavailable</AlertTitle>
            <AlertDescription>
              Could not load AI analysis. The stock may not be in the pipeline
              yet.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const sections = data?.sections;

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Lightbulb className="h-5 w-5 text-yellow-500" />
          AI-Powered Analysis
        </CardTitle>
        <CardDescription>
          Educational insights and investment thesis for {ticker}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {/* LLM Toggle Button */}
        {!llmEnabled && (
          <div className="mb-4 p-3 rounded-lg border border-dashed bg-gradient-to-r from-blue-50/50 via-purple-50/50 to-pink-50/50 dark:from-blue-950/20 dark:via-purple-950/20 dark:to-pink-950/20 border-blue-200/50 dark:border-purple-800/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm">
                <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <defs>
                    <linearGradient
                      id="gemini-cta"
                      x1="0%"
                      y1="0%"
                      x2="100%"
                      y2="100%"
                    >
                      <stop offset="0%" stopColor="#4285F4" />
                      <stop offset="50%" stopColor="#9B72CB" />
                      <stop offset="100%" stopColor="#D96570" />
                    </linearGradient>
                  </defs>
                  <path
                    d="M12 2L14.5 9.5L22 12L14.5 14.5L12 22L9.5 14.5L2 12L9.5 9.5L12 2Z"
                    fill="url(#gemini-cta)"
                  />
                </svg>
                <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent font-medium">
                  Get AI-powered analysis from Gemini
                </span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleEnableLLM}
                className="gap-1.5 border-purple-200 dark:border-purple-700 hover:bg-purple-50 dark:hover:bg-purple-950/50"
              >
                <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
                  <path
                    d="M12 2L14.5 9.5L22 12L14.5 14.5L12 22L9.5 14.5L2 12L9.5 9.5L12 2Z"
                    fill="url(#gemini-cta)"
                  />
                </svg>
                <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Generate
                </span>
              </Button>
            </div>
          </div>
        )}

        <Tabs
          value={activeTab}
          onValueChange={(v) =>
            setActiveTab(v as "ai" | "thesis" | "diagnosis" | "education")
          }
        >
          <TabsList
            className={cn(
              "grid w-full",
              llmEnabled ? "grid-cols-4" : "grid-cols-3"
            )}
          >
            {llmEnabled && (
              <TabsTrigger value="ai" className="flex items-center gap-1.5">
                <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none">
                  <defs>
                    <linearGradient
                      id="gemini-tab"
                      x1="0%"
                      y1="0%"
                      x2="100%"
                      y2="100%"
                    >
                      <stop offset="0%" stopColor="#4285F4" />
                      <stop offset="50%" stopColor="#9B72CB" />
                      <stop offset="100%" stopColor="#D96570" />
                    </linearGradient>
                  </defs>
                  <path
                    d="M12 2L14.5 9.5L22 12L14.5 14.5L12 22L9.5 14.5L2 12L9.5 9.5L12 2Z"
                    fill="url(#gemini-tab)"
                  />
                </svg>
                <span className="hidden sm:inline">AI Summary</span>
              </TabsTrigger>
            )}
            <TabsTrigger value="thesis" className="flex items-center gap-1">
              <Scale className="h-4 w-4" />
              <span className="hidden sm:inline">Should I Buy?</span>
            </TabsTrigger>
            <TabsTrigger value="diagnosis" className="flex items-center gap-1">
              <Stethoscope className="h-4 w-4" />
              <span className="hidden sm:inline">Deep Dive</span>
            </TabsTrigger>
            <TabsTrigger value="education" className="flex items-center gap-1">
              <GraduationCap className="h-4 w-4" />
              <span className="hidden sm:inline">Basics</span>
            </TabsTrigger>
          </TabsList>

          {llmEnabled && (
            <TabsContent value="ai" className="mt-4">
              <LLMSummaryTab
                data={llmData?.sections?.llm_summary}
                isLoading={llmLoading}
              />
            </TabsContent>
          )}

          <TabsContent value="thesis" className="mt-4">
            {sections?.investment_thesis && (
              <ThesisTab thesis={sections.investment_thesis} />
            )}
          </TabsContent>

          <TabsContent value="diagnosis" className="mt-4">
            {sections?.diagnosis && (
              <DiagnosisTab diagnosis={sections.diagnosis} />
            )}
          </TabsContent>

          <TabsContent value="education" className="mt-4">
            {sections?.education && (
              <EducationTab education={sections.education} />
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

// =============================================================================
// Sub-components
// =============================================================================

function EducationTab({ education }: { education: EducationSection }) {
  if (!education.available) {
    return (
      <Alert>
        <AlertDescription>{education.message}</AlertDescription>
      </Alert>
    );
  }

  return (
    <Accordion type="single" collapsible className="w-full">
      {/* Fair Value Explained */}
      {education.what_is_fair_value && (
        <AccordionItem value="fair-value">
          <AccordionTrigger className="text-left">
            <div className="flex items-center gap-2">
              <Target className="h-4 w-4 text-blue-500" />
              {education.what_is_fair_value.title}
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <p className="text-muted-foreground whitespace-pre-line">
              {education.what_is_fair_value.content}
            </p>
            <Badge
              variant="outline"
              className={cn(
                "mt-3",
                education.what_is_fair_value.key_insight === "undervalued"
                  ? "border-green-500 text-green-700"
                  : education.what_is_fair_value.key_insight === "overvalued"
                  ? "border-red-500 text-red-700"
                  : "border-yellow-500 text-yellow-700"
              )}
            >
              {education.what_is_fair_value.key_insight
                .replace("_", " ")
                .toUpperCase()}
            </Badge>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* Pillar Scores Explained */}
      {education.pillar_scores_explained && (
        <AccordionItem value="pillars">
          <AccordionTrigger className="text-left">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-purple-500" />
              {education.pillar_scores_explained.title}
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="grid gap-3">
              {Object.entries(education.pillar_scores_explained.pillars).map(
                ([key, pillar]) => (
                  <div
                    key={key}
                    className="p-3 rounded-lg bg-muted/50 border border-muted"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium capitalize">{key}</span>
                      <span
                        className={cn(
                          "font-bold",
                          getPillarColor(pillar.score)
                        )}
                      >
                        {pillar.score.toFixed(0)}
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {pillar.meaning}
                    </p>
                    <p className="text-xs text-muted-foreground/70 mt-1">
                      Measures: {pillar.what_it_measures}
                    </p>
                  </div>
                )
              )}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* Classification Explained */}
      {education.classification_explained && (
        <AccordionItem value="classification">
          <AccordionTrigger className="text-left">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-yellow-500" />
              {education.classification_explained.title}
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="mb-3">
              <span className="text-sm text-muted-foreground">
                Current Classification:
              </span>
              <Badge variant="outline" className="ml-2">
                {education.classification_explained.current}
              </Badge>
            </div>
            <div className="space-y-2">
              {Object.entries(education.classification_explained.meanings).map(
                ([key, meaning]) => (
                  <div
                    key={key}
                    className={cn(
                      "p-2 rounded text-sm",
                      key.toLowerCase() ===
                        education.classification_explained?.current.toLowerCase()
                        ? "bg-primary/10 border border-primary/20"
                        : "bg-muted/30"
                    )}
                  >
                    <span className="font-medium">{key}:</span> {meaning}
                  </div>
                )
              )}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* Risk Education */}
      {education.risk_education && (
        <AccordionItem value="risk">
          <AccordionTrigger className="text-left">
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-red-500" />
              {education.risk_education.title}
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <p className="text-muted-foreground mb-3">
              {education.risk_education.beta_explained}
            </p>
            <Alert>
              <Lightbulb className="h-4 w-4" />
              <AlertDescription>
                {education.risk_education.diversification_tip}
              </AlertDescription>
            </Alert>
          </AccordionContent>
        </AccordionItem>
      )}
    </Accordion>
  );
}

function DiagnosisTab({ diagnosis }: { diagnosis: DiagnosisSection }) {
  if (!diagnosis.available) {
    return (
      <Alert>
        <AlertDescription>{diagnosis.message}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-4">
      {/* Overall Assessment */}
      {diagnosis.overall_assessment && (
        <div className="p-4 rounded-lg bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/30 dark:to-purple-950/30 border">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Composite Score</span>
            <span
              className={cn(
                "text-2xl font-bold",
                diagnosis.overall_assessment.composite_score >= 70
                  ? "text-green-600"
                  : diagnosis.overall_assessment.composite_score >= 50
                  ? "text-yellow-600"
                  : "text-red-600"
              )}
            >
              {diagnosis.overall_assessment.composite_score.toFixed(0)}
            </span>
          </div>
          <p className="text-sm text-muted-foreground">
            {diagnosis.overall_assessment.interpretation}
          </p>
          <div className="flex items-center gap-2 mt-2">
            <Badge
              variant="outline"
              className={cn(
                diagnosis.overall_assessment.valuation_status === "undervalued"
                  ? "border-green-500 text-green-700"
                  : diagnosis.overall_assessment.valuation_status ===
                    "overvalued"
                  ? "border-red-500 text-red-700"
                  : "border-yellow-500 text-yellow-700"
              )}
            >
              {diagnosis.overall_assessment.margin_of_safety > 0 ? "+" : ""}
              {diagnosis.overall_assessment.margin_of_safety.toFixed(1)}% MoS
            </Badge>
          </div>
        </div>
      )}

      {/* Key Metrics Snapshot */}
      {diagnosis.metrics_snapshot && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-3 rounded-lg bg-slate-50 dark:bg-slate-900/50 border">
          {diagnosis.metrics_snapshot.valuation?.current_price && (
            <div className="text-center">
              <div className="text-xs text-muted-foreground uppercase">
                Price
              </div>
              <div className="font-bold text-lg">
                ${diagnosis.metrics_snapshot.valuation.current_price.toFixed(2)}
              </div>
            </div>
          )}
          {diagnosis.metrics_snapshot.valuation?.pe_ratio && (
            <div className="text-center">
              <div className="text-xs text-muted-foreground uppercase">P/E</div>
              <div className="font-bold text-lg">
                {diagnosis.metrics_snapshot.valuation.pe_ratio.toFixed(1)}x
              </div>
            </div>
          )}
          {diagnosis.metrics_snapshot.fundamentals?.roe_pct && (
            <div className="text-center">
              <div className="text-xs text-muted-foreground uppercase">ROE</div>
              <div className="font-bold text-lg">
                {diagnosis.metrics_snapshot.fundamentals.roe_pct.toFixed(1)}%
              </div>
            </div>
          )}
          {diagnosis.metrics_snapshot.risk?.beta && (
            <div className="text-center">
              <div className="text-xs text-muted-foreground uppercase">
                Beta
              </div>
              <div className="font-bold text-lg">
                {diagnosis.metrics_snapshot.risk.beta.toFixed(2)}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Strengths */}
      {diagnosis.strengths && diagnosis.strengths.count > 0 && (
        <div>
          <h4 className="font-medium flex items-center gap-2 mb-2">
            <CheckCircle2 className="h-4 w-4 text-green-500" />
            Strengths ({diagnosis.strengths.count})
          </h4>
          <div className="space-y-2">
            {diagnosis.strengths.pillars.map((pillar, i) => (
              <div
                key={i}
                className="p-3 rounded-lg bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-800"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">{pillar.name}</span>
                  <span className="text-green-600 font-bold">
                    {pillar.score.toFixed(0)}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground mt-1">
                  {pillar.why_strong}
                </p>
                {/* Supporting Metrics */}
                {pillar.supporting_metrics &&
                  pillar.supporting_metrics.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-2 pt-2 border-t border-green-200 dark:border-green-800">
                      {pillar.supporting_metrics.map((metric, j) => (
                        <span
                          key={j}
                          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300"
                          title={metric.note}
                        >
                          <span className="font-medium">{metric.name}:</span>
                          <span>{metric.value}</span>
                          {metric.note && (
                            <span className="text-green-600 dark:text-green-400">
                              ({metric.note})
                            </span>
                          )}
                        </span>
                      ))}
                    </div>
                  )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Weaknesses */}
      {diagnosis.weaknesses && diagnosis.weaknesses.count > 0 && (
        <div>
          <h4 className="font-medium flex items-center gap-2 mb-2">
            <XCircle className="h-4 w-4 text-red-500" />
            Weaknesses ({diagnosis.weaknesses.count})
          </h4>
          <div className="space-y-2">
            {diagnosis.weaknesses.pillars.map((pillar, i) => (
              <div
                key={i}
                className="p-3 rounded-lg bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800"
              >
                <div className="flex items-center justify-between">
                  <span className="font-medium">{pillar.name}</span>
                  <span className="text-red-600 font-bold">
                    {pillar.score.toFixed(0)}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground mt-1">
                  {pillar.concern}
                </p>
                {/* Supporting Metrics */}
                {pillar.supporting_metrics &&
                  pillar.supporting_metrics.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-2 pt-2 border-t border-red-200 dark:border-red-800">
                      {pillar.supporting_metrics.map((metric, j) => (
                        <span
                          key={j}
                          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300"
                          title={metric.note}
                        >
                          <span className="font-medium">{metric.name}:</span>
                          <span>{metric.value}</span>
                          {metric.note && (
                            <span className="text-red-600 dark:text-red-400">
                              ({metric.note})
                            </span>
                          )}
                        </span>
                      ))}
                    </div>
                  )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Key Drivers */}
      {diagnosis.key_drivers && diagnosis.key_drivers.length > 0 && (
        <div>
          <h4 className="font-medium flex items-center gap-2 mb-2">
            <Zap className="h-4 w-4 text-yellow-500" />
            Key Value Drivers
          </h4>
          <ul className="space-y-1">
            {diagnosis.key_drivers.map((driver, i) => (
              <li
                key={i}
                className="text-sm text-muted-foreground flex items-start gap-2"
              >
                <span className="text-yellow-500 mt-0.5">•</span>
                {driver}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Red Flags */}
      {diagnosis.red_flags && diagnosis.red_flags.length > 0 && (
        <Alert variant="destructive" className="border-red-200">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Red Flags</AlertTitle>
          <AlertDescription>
            <ul className="mt-2 space-y-1">
              {diagnosis.red_flags.map((flag, i) => (
                <li key={i} className="text-sm">
                  • {flag}
                </li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {/* Catalysts */}
      {diagnosis.catalysts && diagnosis.catalysts.length > 0 && (
        <div>
          <h4 className="font-medium flex items-center gap-2 mb-2">
            <TrendingUp className="h-4 w-4 text-blue-500" />
            Potential Catalysts
          </h4>
          <ul className="space-y-1">
            {diagnosis.catalysts.map((catalyst, i) => (
              <li
                key={i}
                className="text-sm text-muted-foreground flex items-start gap-2"
              >
                <span className="text-blue-500 mt-0.5">→</span>
                {catalyst}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function ThesisTab({ thesis }: { thesis: InvestmentThesisSection }) {
  if (!thesis.available) {
    return (
      <Alert>
        <AlertDescription>{thesis.message}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header with recommendation */}
      <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50 border">
        <div>
          <span className="text-sm text-muted-foreground">Recommendation</span>
          <div className="flex items-center gap-2 mt-1">
            <Badge
              className={cn(
                thesis.recommendation === "Buy"
                  ? "bg-green-500"
                  : thesis.recommendation === "Hold"
                  ? "bg-blue-500"
                  : "bg-yellow-500"
              )}
            >
              {thesis.recommendation}
            </Badge>
            <span className="text-sm text-muted-foreground">
              Conviction: {thesis.conviction}
            </span>
          </div>
        </div>
      </div>

      {/* Bull Case */}
      {thesis.bull_case && (
        <div className="p-4 rounded-lg bg-green-50 dark:bg-green-950/20 border border-green-200 dark:border-green-800">
          <h4 className="font-medium flex items-center gap-2 mb-2 text-green-700 dark:text-green-400">
            <TrendingUp className="h-4 w-4" />
            Bull Case
          </h4>
          <p className="text-sm text-muted-foreground mb-3">
            {thesis.bull_case.summary}
          </p>
          <ul className="space-y-1">
            {thesis.bull_case.points.map((point, i) => (
              <li key={i} className="text-sm flex items-start gap-2">
                <CheckCircle2 className="h-3 w-3 text-green-500 mt-1 flex-shrink-0" />
                {point}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Bear Case */}
      {thesis.bear_case && (
        <div className="p-4 rounded-lg bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800">
          <h4 className="font-medium flex items-center gap-2 mb-2 text-red-700 dark:text-red-400">
            <TrendingDown className="h-4 w-4" />
            Bear Case
          </h4>
          <p className="text-sm text-muted-foreground mb-3">
            {thesis.bear_case.summary}
          </p>
          <ul className="space-y-1">
            {thesis.bear_case.points.map((point, i) => (
              <li key={i} className="text-sm flex items-start gap-2">
                <AlertTriangle className="h-3 w-3 text-red-500 mt-1 flex-shrink-0" />
                {point}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Base Case */}
      {thesis.base_case && (
        <div className="p-4 rounded-lg bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800">
          <h4 className="font-medium flex items-center gap-2 mb-2 text-blue-700 dark:text-blue-400">
            <Target className="h-4 w-4" />
            Base Case
          </h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Expected Return</span>
              <p className="font-medium">{thesis.base_case.expected_return}</p>
            </div>
            <div>
              <span className="text-muted-foreground">Timeframe</span>
              <p className="font-medium">{thesis.base_case.timeframe}</p>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Key assumption: {thesis.base_case.key_assumption}
          </p>
        </div>
      )}

      {/* Action Items */}
      {thesis.action_items && thesis.action_items.length > 0 && (
        <div>
          <h4 className="font-medium flex items-center gap-2 mb-2">
            <Lightbulb className="h-4 w-4 text-yellow-500" />
            Action Items
          </h4>
          <ul className="space-y-1">
            {thesis.action_items.map((item, i) => (
              <li
                key={i}
                className="text-sm text-muted-foreground flex items-start gap-2"
              >
                <span className="text-primary font-medium">{i + 1}.</span>
                {item}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// Simple markdown parser for the AI response
function parseMarkdownToSections(
  text: string
): { title: string; content: string }[] {
  if (!text) return [];

  // Helper to strip emojis from text
  const stripEmojis = (str: string) => {
    return str
      .replace(/[\u{1F300}-\u{1F9FF}]/gu, "") // Misc symbols and pictographs
      .replace(/[\u{2600}-\u{26FF}]/gu, "") // Misc symbols
      .replace(/[\u{2700}-\u{27BF}]/gu, "") // Dingbats
      .replace(/[\u{1F600}-\u{1F64F}]/gu, "") // Emoticons
      .replace(/[\u{1F680}-\u{1F6FF}]/gu, "") // Transport and map
      .replace(/[\u{1F1E0}-\u{1F1FF}]/gu, "") // Flags
      .replace(/[\u{2300}-\u{23FF}]/gu, "") // Tech symbols
      .replace(/[\u{2B50}]/gu, "") // Star
      .replace(/[\u{FE00}-\u{FE0F}]/gu, "") // Variation selectors
      .trim();
  };

  const sections: { title: string; content: string }[] = [];
  const lines = text.split("\n");
  let currentSection: { title: string; content: string } | null = null;

  for (const line of lines) {
    // Match headers like "## TLDR" or "## 💪 Key Strength"
    const headerMatch = line.match(/^##\s*(.+)$/);
    if (headerMatch) {
      if (currentSection) {
        sections.push(currentSection);
      }
      // Strip emojis and clean up title
      const cleanTitle = stripEmojis(headerMatch[1].trim());
      currentSection = {
        title: cleanTitle,
        content: "",
      };
    } else if (currentSection) {
      // Add content to current section (also strip random emojis from content for consistency)
      const cleanLine = line;
      currentSection.content +=
        (currentSection.content ? "\n" : "") + cleanLine;
    }
  }

  if (currentSection) {
    sections.push(currentSection);
  }

  return sections;
}

// LLM Summary Tab - Shows Gemini-generated analysis with typewriter effect
function LLMSummaryTab({
  data,
  isLoading,
}: {
  data?: LLMSummarySection;
  isLoading: boolean;
}) {
  const [showTypewriter, setShowTypewriter] = useState(true);

  if (isLoading) {
    return (
      <div className="space-y-6 py-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            {/* Gemini-style sparkle icon */}
            <svg className="h-6 w-6" viewBox="0 0 24 24" fill="none">
              <defs>
                <linearGradient
                  id="gemini-gradient"
                  x1="0%"
                  y1="0%"
                  x2="100%"
                  y2="100%"
                >
                  <stop offset="0%" stopColor="#4285F4" />
                  <stop offset="25%" stopColor="#9B72CB" />
                  <stop offset="50%" stopColor="#D96570" />
                  <stop offset="75%" stopColor="#D96570" />
                  <stop offset="100%" stopColor="#9B72CB" />
                </linearGradient>
              </defs>
              <path
                d="M12 2L14.5 9.5L22 12L14.5 14.5L12 22L9.5 14.5L2 12L9.5 9.5L12 2Z"
                fill="url(#gemini-gradient)"
                className="animate-pulse"
              />
            </svg>
            <span className="absolute -top-1 -right-1 flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-gradient-to-r from-blue-500 to-purple-500"></span>
            </span>
          </div>
          <span className="text-sm font-medium bg-gradient-to-r from-blue-600 via-purple-500 to-pink-500 bg-clip-text text-transparent">
            Gemini is analyzing this stock...
          </span>
        </div>
        <div className="space-y-4">
          <Skeleton className="h-16 w-full rounded-lg" />
          <Skeleton className="h-24 w-3/4 rounded-lg" />
          <Skeleton className="h-20 w-full rounded-lg" />
          <Skeleton className="h-20 w-5/6 rounded-lg" />
        </div>
      </div>
    );
  }

  if (!data?.available) {
    return (
      <Alert>
        <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none">
          <path
            d="M12 2L14.5 9.5L22 12L14.5 14.5L12 22L9.5 14.5L2 12L9.5 9.5L12 2Z"
            fill="currentColor"
          />
        </svg>
        <AlertTitle>AI Analysis Unavailable</AlertTitle>
        <AlertDescription>
          {data?.message ||
            "Could not generate AI analysis for this stock. Try again later."}
        </AlertDescription>
      </Alert>
    );
  }

  const sections = parseMarkdownToSections(data.analysis || "");

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Badge
            variant="secondary"
            className="gap-1.5 px-2.5 py-1 bg-gradient-to-r from-blue-50 via-purple-50 to-pink-50 dark:from-blue-950/40 dark:via-purple-950/40 dark:to-pink-950/40 border border-blue-200/50 dark:border-purple-700/50"
          >
            <svg className="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
              <defs>
                <linearGradient
                  id="gemini-badge"
                  x1="0%"
                  y1="0%"
                  x2="100%"
                  y2="100%"
                >
                  <stop offset="0%" stopColor="#4285F4" />
                  <stop offset="50%" stopColor="#9B72CB" />
                  <stop offset="100%" stopColor="#D96570" />
                </linearGradient>
              </defs>
              <path
                d="M12 2L14.5 9.5L22 12L14.5 14.5L12 22L9.5 14.5L2 12L9.5 9.5L12 2Z"
                fill="url(#gemini-badge)"
              />
            </svg>
            <span className="text-xs font-medium bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
              {data.generated_by === "gemini"
                ? "Gemini AI"
                : data.generated_by || "AI Generated"}
            </span>
          </Badge>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowTypewriter(!showTypewriter)}
          className="text-xs text-muted-foreground"
        >
          {showTypewriter ? "Skip animation" : "Replay"}
        </Button>
      </div>

      {/* Parsed Sections with nice cards */}
      <div className="space-y-3">
        {sections.map((section, idx) => (
          <AISection
            key={idx}
            title={section.title}
            content={section.content}
            index={idx}
            animate={showTypewriter}
          />
        ))}
      </div>

      {/* Context Footer */}
      {data.data_context && (
        <div className="pt-4 mt-4 border-t border-dashed">
          <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
            {data.data_context.fair_value &&
              data.data_context.fair_value > 0 && (
                <span className="flex items-center gap-1">
                  <Target className="h-3 w-3" />
                  Fair Value:{" "}
                  <span className="font-medium text-foreground">
                    ${data.data_context.fair_value.toFixed(2)}
                  </span>
                </span>
              )}
            {data.data_context.current_price &&
              data.data_context.current_price > 0 && (
                <span className="flex items-center gap-1">
                  Price:{" "}
                  <span className="font-medium text-foreground">
                    ${data.data_context.current_price.toFixed(2)}
                  </span>
                </span>
              )}
            {data.data_context.consensus_score && (
              <span className="flex items-center gap-1">
                Score:{" "}
                <span className="font-medium text-foreground">
                  {data.data_context.consensus_score.toFixed(0)}/100
                </span>
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Helper function to parse inline markdown (bold, italic, code)
function parseInlineMarkdown(text: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  let key = 0;

  // Pattern for **bold**, *italic*, `code`
  const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g;
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    // Add text before the match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    // Determine which pattern matched
    if (match[2]) {
      // **bold**
      parts.push(
        <strong key={key++} className="font-semibold text-foreground">
          {match[2]}
        </strong>
      );
    } else if (match[3]) {
      // *italic*
      parts.push(
        <em key={key++} className="italic">
          {match[3]}
        </em>
      );
    } else if (match[4]) {
      // `code`
      parts.push(
        <code
          key={key++}
          className="px-1 py-0.5 rounded bg-muted text-xs font-mono"
        >
          {match[4]}
        </code>
      );
    }

    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length > 0 ? parts : [text];
}

// Individual AI section card with optional typewriter effect
function AISection({
  title,
  content,
  index,
  animate,
}: {
  title: string;
  content: string;
  index: number;
  animate: boolean;
}) {
  const [visible, setVisible] = useState(!animate);

  // Stagger animation for each section
  useEffect(() => {
    if (!animate) return;
    const timer = setTimeout(() => setVisible(true), index * 200);
    return () => clearTimeout(timer);
  }, [animate, index]);

  // Determine section styling based on title
  const getSectionStyle = () => {
    const lowerTitle = title.toLowerCase();
    if (lowerTitle.includes("tldr") || lowerTitle.includes("summary")) {
      return "bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-950/30 dark:to-yellow-950/30 border-amber-200 dark:border-amber-800";
    }
    if (lowerTitle.includes("strength") || lowerTitle.includes("key")) {
      return "bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950/30 dark:to-emerald-950/30 border-green-200 dark:border-green-800";
    }
    if (lowerTitle.includes("risk") || lowerTitle.includes("warning")) {
      return "bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-950/30 dark:to-orange-950/30 border-red-200 dark:border-red-800";
    }
    if (lowerTitle.includes("do") || lowerTitle.includes("action")) {
      return "bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 border-blue-200 dark:border-blue-800";
    }
    return "bg-muted/50 border-border";
  };

  const getIcon = () => {
    // Always use consistent Lucide icons based on title keywords - never emojis
    const lowerTitle = title.toLowerCase();

    // TLDR / Summary
    if (
      lowerTitle.includes("tldr") ||
      lowerTitle.includes("summary") ||
      lowerTitle.includes("overview")
    )
      return <Zap className="h-4 w-4 text-amber-600" />;

    // Strengths / Positives / Bull
    if (
      lowerTitle.includes("strength") ||
      lowerTitle.includes("positive") ||
      lowerTitle.includes("bull") ||
      lowerTitle.includes("pro")
    )
      return <CheckCircle2 className="h-4 w-4 text-green-600" />;

    // Risks / Warnings / Bear / Concerns
    if (
      lowerTitle.includes("risk") ||
      lowerTitle.includes("warning") ||
      lowerTitle.includes("bear") ||
      lowerTitle.includes("concern") ||
      lowerTitle.includes("weak")
    )
      return <AlertTriangle className="h-4 w-4 text-red-600" />;

    // Actions / What to do / Recommendation
    if (
      lowerTitle.includes("do") ||
      lowerTitle.includes("action") ||
      lowerTitle.includes("recommend") ||
      lowerTitle.includes("next")
    )
      return <Target className="h-4 w-4 text-blue-600" />;

    // Growth / Opportunity
    if (
      lowerTitle.includes("growth") ||
      lowerTitle.includes("opportunity") ||
      lowerTitle.includes("upside")
    )
      return <TrendingUp className="h-4 w-4 text-emerald-600" />;

    // Valuation / Value
    if (
      lowerTitle.includes("valuation") ||
      lowerTitle.includes("value") ||
      lowerTitle.includes("price")
    )
      return <BarChart3 className="h-4 w-4 text-purple-600" />;

    // Quality / Fundamentals
    if (lowerTitle.includes("quality") || lowerTitle.includes("fundamental"))
      return <Shield className="h-4 w-4 text-indigo-600" />;

    // Default
    return <Lightbulb className="h-4 w-4 text-muted-foreground" />;
  };

  // Parse content lines and render with markdown support
  const renderContent = () => {
    const lines = content.trim().split("\n");
    return lines.map((line, i) => {
      const trimmedLine = line.trim();
      // Check if it's a bullet point
      if (trimmedLine.startsWith("- ") || trimmedLine.startsWith("• ")) {
        return (
          <li key={i} className="flex items-start gap-2 ml-2">
            <span className="text-primary mt-1">•</span>
            <span>{parseInlineMarkdown(trimmedLine.slice(2))}</span>
          </li>
        );
      }
      // Check if it's a numbered list
      const numberedMatch = trimmedLine.match(/^(\d+)\.\s+(.+)$/);
      if (numberedMatch) {
        return (
          <li key={i} className="flex items-start gap-2 ml-2">
            <span className="text-primary font-medium">
              {numberedMatch[1]}.
            </span>
            <span>{parseInlineMarkdown(numberedMatch[2])}</span>
          </li>
        );
      }
      // Regular line
      if (trimmedLine) {
        return (
          <p key={i} className={i > 0 ? "mt-1" : ""}>
            {parseInlineMarkdown(trimmedLine)}
          </p>
        );
      }
      return null;
    });
  };

  if (!visible && animate) {
    return <Skeleton className="h-20 w-full rounded-lg" />;
  }

  return (
    <div
      className={cn(
        "rounded-lg border p-4 transition-all duration-300",
        getSectionStyle(),
        animate && "animate-in fade-in-0 slide-in-from-bottom-2"
      )}
      style={{ animationDelay: `${index * 100}ms` }}
    >
      <div className="flex items-start gap-3">
        <div className="mt-0.5">{getIcon()}</div>
        <div className="flex-1 min-w-0">
          <h4 className="font-semibold text-sm mb-1">{title}</h4>
          <div className="text-sm text-muted-foreground leading-relaxed space-y-0.5">
            {renderContent()}
          </div>
        </div>
      </div>
    </div>
  );
}
