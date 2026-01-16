"use client";

import React, { useCallback, memo, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Sun, Moon, Menu, MapPin } from "lucide-react";
import { motion } from "framer-motion";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Badge } from "@/components/ui/badge";
import { City } from "@/types/weather";
import LogoSvg from "../../public/svg/logo";
import HeaderProfile from "@/app/dashboard/_components/header-profile";

const Logo = memo(({ onClick }: { onClick: () => void }) => (
  <Button
    variant="ghost"
    onClick={onClick}
    className="flex-shrink-0 flex flex-row items-center justify-between gap-4 group hover:scale-105 active:scale-95 transition-transform duration-200 h-auto p-0 hover:bg-transparent"
    aria-label="logo"
  >
    <div className="relative">
      <div className="absolute inset-0 bg-gradient-to-br from-teal-500 to-emerald-600 rounded-full blur-lg opacity-20 group-hover:opacity-40 transition-opacity duration-300" />
      <div className="relative p-3 bg-gradient-to-br from-teal-600 to-emerald-700 rounded-full shadow-lg">
        <LogoSvg className="w-8 h-8 md:w-10 md:h-10 text-white" fill="currentColor" />
      </div>
    </div>
    <div className="hidden sm:block text-left">
      <div className="text-xl md:text-2xl font-bold bg-gradient-to-r from-teal-600 via-emerald-600 to-green-700 bg-clip-text text-transparent">
        ZonaPETIK
      </div>
      <div className="text-sm text-gray-500 dark:text-gray-400 -mt-1">
        Weather Dashboard
      </div>
    </div>
  </Button>
));
Logo.displayName = "Logo";

const SearchSuggestions = memo(({ suggestions, selectedIndex, onSelect, onMouseEnter }: {
  suggestions: City[];
  selectedIndex: number;
  onSelect: (city: City) => void;
  onMouseEnter: (index: number) => void
}) => (
  <div className="absolute left-0 right-0 top-full mt-2 z-50 animate-slideDown">
    <div className="bg-white dark:bg-gray-800 shadow-2xl rounded-2xl border border-gray-200 dark:border-gray-700 backdrop-blur-lg">
      <ul className="max-h-60 rounded-2xl py-2 overflow-auto" role="listbox" aria-label="City suggestions" tabIndex={-1}>
        {suggestions.map((city, index) => (
          <li
            key={`${city.name}-${city.lat}-${city.lon}`}
            className={`cursor-pointer select-none relative py-3 px-4 mx-2 rounded-xl transition-all duration-200 hover:bg-teal-50 dark:hover:bg-teal-900/20 ${
              index === selectedIndex ? "bg-teal-50 dark:bg-teal-900/20 border-l-4 border-teal-600" : ""
            }`}
            onClick={() => onSelect(city)}
            onMouseEnter={() => onMouseEnter(index)}
            role="option"
            aria-selected={index === selectedIndex}
            tabIndex={-1}
            id={`city-option-${index}`}
          >
            <div className="flex items-center gap-3">
              <MapPin className="w-4 h-4 text-gray-400" />
              <span className="block truncate font-medium text-gray-800 dark:text-gray-200">
                {city.name}
              </span>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {city.country}
              </span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  </div>
));
SearchSuggestions.displayName = "SearchSuggestions";

const WeatherStatus = memo(() => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [isDaytime, setIsDaytime] = useState(true);

  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setCurrentTime(now);
      const hours = now.getHours();
      setIsDaytime(hours >= 6 && hours < 18);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return (
    <Badge 
      variant="outline" 
      className="hidden md:flex items-center gap-3 px-4 py-2 bg-gradient-to-r from-teal-50 to-emerald-50 dark:from-teal-900/20 dark:to-emerald-900/20 border-teal-200 dark:border-teal-700 animate-fadeIn"
    >
      <div className="transition-transform duration-300 hover:scale-110">
        {isDaytime ? (
          <Sun className="w-4 h-4 text-yellow-500" />
        ) : (
          <Moon className="w-4 h-4 text-blue-400" />
        )}
      </div>
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300 whitespace-nowrap">
        {currentTime.toLocaleTimeString('id-ID', {
          hour: '2-digit',
          minute: '2-digit'
        })}
      </span>
    </Badge>
  );
});
WeatherStatus.displayName = "WeatherStatus";

const MobileMenu = memo(() => {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" className="md:hidden">
          <Menu className="h-5 w-5" />
          <span className="sr-only">Toggle menu</span>
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="w-[300px] sm:w-[400px]">
        <div className="flex flex-col gap-4 py-4">
          <div className="space-y-2">
            <h2 className="text-lg font-semibold">Navigation</h2>
            <Separator />
          </div>
          <WeatherStatus />
        </div>
      </SheetContent>
    </Sheet>
  );
});
MobileMenu.displayName = "MobileMenu";

const NavBar = () => {
  const router = useRouter();
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleLogoClick = useCallback(() => {
    router.refresh();
  }, [router]);

  return (
    <motion.nav
      initial={{ y: -10, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.3 }}
      className={`
        sticky top-0 z-50 transition-all duration-300 backdrop-blur-xl
        ${isScrolled
          ? 'bg-white/90 dark:bg-gray-900/90 shadow-lg border-b border-gray-200/60 dark:border-gray-700/60'
          : 'bg-white/70 dark:bg-gray-900/70 border-b border-gray-200/40 dark:border-gray-700/40'
        }
      `}
      role="navigation"
      aria-label="Main navigation"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 lg:h-20">
          
          {/* Left Section - Logo */}
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="flex items-center gap-4 lg:gap-6 flex-1 min-w-0"
          >
            <Logo onClick={handleLogoClick} />
          </motion.div>

          {/* Center Section - Weather Status */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6 }}
            className="flex-shrink-0"
          >
            <WeatherStatus />
          </motion.div>

          {/* Right Section - Profile & Mobile Menu */}
          <motion.div
            initial={{ x: 20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="flex items-center justify-end gap-2 flex-1 min-w-0"
          >
            <div className="relative hover:scale-105 transition-transform duration-200">
              <HeaderProfile />
            </div>
            <MobileMenu />
          </motion.div>

        </div>
      </div>

      {/* Progress Bar */}
      <motion.div
        className="absolute bottom-0 left-0 h-1 bg-gradient-to-r from-teal-500 via-emerald-600 to-green-600 rounded-full"
        initial={{ width: "0%" }}
        animate={{ width: isScrolled ? "100%" : "0%" }}
        transition={{ duration: 0.3, ease: "easeOut" }}
      />

      <style jsx>{`
        .animate-fadeIn {
          opacity: 0;
          animation: fadeIn 0.6s ease-out forwards;
        }
        
        @keyframes fadeIn {
          to {
            opacity: 1;
          }
        }
      `}</style>
    </motion.nav>
  );
};

export default memo(NavBar);
