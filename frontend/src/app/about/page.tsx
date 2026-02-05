/**
 * About Page - Project Information
 *
 * Information about Emetix, the AI-powered stock screening platform.
 */
"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  TrendingUp,
  Brain,
  Shield,
  Target,
  Sparkles,
  BarChart3,
  LineChart,
} from "lucide-react";

export default function AboutPage() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-8 max-w-4xl">
      {/* Hero Section */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-3">
          <TrendingUp className="h-12 w-12 text-primary" />
          <h1 className="text-4xl font-bold">Emetix</h1>
        </div>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          AI-Powered Multi-Agent Low-Risk Stock Watchlist & Risk Management
          Platform
        </p>
        <div className="flex flex-wrap gap-2 justify-center">
          <Badge variant="secondary">AI-Driven Valuation</Badge>
          <Badge variant="outline">ML-Powered</Badge>
          <Badge variant="outline">UnderValued Hunter</Badge>
        </div>
      </div>

      {/* Mission */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5 text-blue-600" />
            Our Mission
          </CardTitle>
        </CardHeader>
        <CardContent className="text-muted-foreground">
          <p>
            Emetix helps retail investors quickly identify quality, low-risk
            investment opportunities through AI-powered analysis. We combine
            traditional financial metrics with cutting-edge machine learning to
            provide comprehensive stock valuations and risk assessments.
          </p>
        </CardContent>
      </Card>

      {/* Key Features */}
      <div className="grid md:grid-cols-2 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-purple-600" />
              3-Stage Screening Pipeline
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            <ul className="space-y-1">
              <li>• Attention triggers for opportunity detection</li>
              <li>• 4-Pillar scoring (Value, Quality, Growth, Safety)</li>
              <li>• Buy/Hold/Watch classification</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Brain className="h-5 w-5 text-pink-600" />
              Multi-Agent AI System
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            <ul className="space-y-1">
              <li>• Sentiment analysis from news & social media</li>
              <li>• Fundamental analysis agent</li>
              <li>• ML-powered fair value estimation</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <LineChart className="h-5 w-5 text-green-600" />
              LSTM-DCF Valuation
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            <ul className="space-y-1">
              <li>• Deep learning growth forecasting</li>
              <li>• Hybrid ML + traditional DCF model</li>
              <li>• Consensus scoring methodology</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Shield className="h-5 w-5 text-amber-600" />
              Personal Risk Capacity
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            <ul className="space-y-1">
              <li>• Customized risk assessment questionnaire</li>
              <li>• Position sizing recommendations</li>
              <li>• Stock-to-profile matching</li>
            </ul>
          </CardContent>
        </Card>
      </div>

      {/* Tech Stack */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 pb-3">
            <BarChart3 className="h-5 w-5 text-indigo-600" />
            Technology Stack
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <h4 className="font-medium mb-2">Frontend</h4>
              <ul className="text-muted-foreground space-y-1">
                <li>Next.js 16</li>
                <li>React 19</li>
                <li>Tailwind CSS</li>
                <li>Recharts</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Backend</h4>
              <ul className="text-muted-foreground space-y-1">
                <li>FastAPI</li>
                <li>Python 3.11+</li>
                <li>MongoDB</li>
                <li>LangChain</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">ML/AI</h4>
              <ul className="text-muted-foreground space-y-1">
                <li>PyTorch</li>
                <li>LSTM Networks</li>
                <li>Google Gemini</li>
                <li>CUDA 11.8</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Data Sources</h4>
              <ul className="text-muted-foreground space-y-1">
                <li>Yahoo Finance</li>
                <li>Alpha Vantage</li>
                <li>Finnhub</li>
                <li>NewsAPI</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Disclaimer */}
      <Card className="border-amber-200 bg-amber-50/50 dark:bg-amber-950/20">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg text-amber-800 dark:text-amber-200">
            Disclaimer
          </CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-amber-700 dark:text-amber-300">
          <p>
            Emetix is an educational project and does not constitute financial
            advice. All stock analysis and recommendations are for informational
            purposes only. Always conduct your own research and consult with a
            qualified financial advisor before making investment decisions.
          </p>
        </CardContent>
      </Card>

      {/* Footer */}
      <div className="text-center text-sm text-muted-foreground pt-4 border-t">
        <p>© 2026 Emetix - A00303759 Final Year Project</p>
        <p className="mt-1">Built with care for retail investors</p>
      </div>
    </div>
  );
}
