/**
 * Navbar Component
 *
 * Main navigation bar for the application.
 * Contains logo, navigation links, theme toggle, and search.
 */
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { TrendingUp, Search, Trophy, User, Filter, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "./ThemeToggle";

const navItems = [
  { href: "/", label: "Top Picks", icon: Trophy },
  { href: "/screener", label: "Stock Screener", icon: Filter },
  { href: "/profile/risk-assessment", label: "Risk Profile", icon: User },
  { href: "/about", label: "About", icon: Info },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="border-b bg-background">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2">
            <TrendingUp className="h-6 w-6 text-primary" />
            <span className="text-xl font-bold">Emetix</span>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;

              return (
                <Link key={item.href} href={item.href}>
                  <Button
                    variant={isActive ? "secondary" : "ghost"}
                    className={cn(
                      "flex items-center gap-2",
                      isActive && "bg-muted",
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    {item.label}
                  </Button>
                </Link>
              );
            })}
          </div>

          {/* Search and Theme Toggle */}
          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon">
              <Search className="h-4 w-4" />
            </Button>
            <ThemeToggle />
          </div>
        </div>
      </div>
    </nav>
  );
}
