"use client";

import React, { useCallback, memo } from "react";
import { City } from "@/types/weather";
import LogoSvg from "../../public/svg/logo";
import { useRouter } from "next/navigation";
import HeaderProfile from "@/app/dashboard/_components/header-profile";

const Logo = memo(({ onClick }: { onClick: () => void }) => (
  <div
    className="flex-shrink-0 flex flex-row items-center justify-between gap-2 cursor-pointer hover:opacity-80 transition-opacity mr-4 md:mr-0"
    onClick={onClick}
    role="button"
    tabIndex={0}
    onKeyDown={(e) => e.key === "Enter" && onClick()}
    aria-label="logo"
  >
    <LogoSvg className="w-8 h-8" fill="currentColor" />
    <span className="text-md md:text-2xl font-bold md:font-bold ">Zona Petik</span>
  </div>
));
Logo.displayName = "Logo";

const SearchSuggestions = memo(({ suggestions, selectedIndex, onSelect, onMouseEnter }: { suggestions: City[]; selectedIndex: number; onSelect: (city: City) => void; onMouseEnter: (index: number) => void }) => (
  <div className="absolute left-0 right-0 top-full mt-1">
    <ul className="w-full bg-background shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm" role="listbox" aria-label="City suggestions" tabIndex={-1}>
      {suggestions.map((city, index) => (
        <li
          key={`${city.name}-${city.lat}-${city.lon}`}
          className={`cursor-pointer select-none relative py-2 pl-3 pr-9 ${index === selectedIndex ? "bg-primary/10" : ""}`}
          onClick={() => onSelect(city)}
          onMouseEnter={() => onMouseEnter(index)}
          role="option"
          aria-selected={index === selectedIndex}
          tabIndex={-1}
          id={`city-option-${index}`}
        >
          <span className="flex items-center">
            <span className="ml-3 block truncate">
              {city.name}, {city.country}
            </span>
          </span>
        </li>
      ))}
    </ul>
  </div>
));
SearchSuggestions.displayName = "SearchSuggestions";

const NavBar = () => {
  const router = useRouter();

  // Memoize handlers
  const handleLogoClick = useCallback(() => {
    router.refresh();
  }, [router]);

  return (
    <nav className="bg-background sticky top-0 z-50 border-b" role="navigation" aria-label="Main navigation">
      <div className="container px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Logo onClick={handleLogoClick} />
          <div className="flex items-center">
            <HeaderProfile />
          </div>
        </div>
      </div>
    </nav>
  );
};

export default memo(NavBar);
