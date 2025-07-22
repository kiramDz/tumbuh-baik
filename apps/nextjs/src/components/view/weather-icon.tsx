export default function WeatherIcon() {
  return (
    <div className="flex items-center justify-center">
      <div className="relative">
        {/* Storm Cloud */}
        <div className="relative">
          {/* Main cloud */}
          <div className="w-48 h-28 bg-gradient-to-br from-gray-300 via-gray-400 to-gray-600 rounded-full opacity-90 shadow-2xl"></div>

          {/* Cloud bumps for more realistic shape */}
          <div className="absolute top-2 left-8 w-20 h-16 bg-gradient-to-br from-gray-200 via-gray-300 to-gray-500 rounded-full opacity-80"></div>
          <div className="absolute top-4 right-6 w-16 h-12 bg-gradient-to-br from-gray-300 via-gray-400 to-gray-600 rounded-full opacity-85"></div>
          <div className="absolute -top-2 left-16 w-24 h-18 bg-gradient-to-br from-white via-gray-200 to-gray-400 rounded-full opacity-75"></div>

          {/* Bright area in cloud */}
          <div className="absolute top-6 left-20 w-16 h-8 bg-gradient-to-r from-yellow-200 via-orange-200 to-yellow-300 rounded-full opacity-60 blur-sm"></div>
        </div>

        {/* Lightning bolt */}
        <div className="absolute top-20 left-20 transform -translate-x-1/2">
          <svg width="40" height="60" viewBox="0 0 40 60" className="drop-shadow-lg">
            <path d="M20 0 L8 22 L16 22 L12 45 L32 18 L24 18 L28 0 Z" fill="url(#lightning-gradient)" stroke="#F59E0B" strokeWidth="1" />
            <defs>
              <linearGradient id="lightning-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#FCD34D" />
                <stop offset="50%" stopColor="#F59E0B" />
                <stop offset="100%" stopColor="#D97706" />
              </linearGradient>
            </defs>
          </svg>
        </div>

        {/* Subtle glow effect */}
        <div className="absolute inset-0 bg-blue-400 opacity-10 rounded-full blur-xl"></div>
      </div>
    </div>
  );
}
