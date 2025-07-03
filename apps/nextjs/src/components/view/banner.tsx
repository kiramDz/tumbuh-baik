import { Badge } from "@/components/ui/badge";

export function Banner() {
  return (
    <div className="relative mb-2 h-96">
      {/* Background Image */}
      <div
        className="absolute inset-0 h-full w-full bg-cover bg-center"
        style={{
          backgroundImage: "url('https://images.unsplash.com/photo-1434725039720-aaad6dd32dfe?q=80&w=2842&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')",
          backgroundBlendMode: "overlay",
        }}
      >
        <div className="absolute inset-0 bg-black/30" />
      </div>

      {/* Content */}
      <div className="relative px-6 py-4 w-1/3">
        <div className="mt-10 rounded-lg bg-white p-6 shadow-lg">
          <div className="mb-6">
            <h1 className="text-2xl font-bold">Aceh Besar</h1>
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <Badge variant="outline" className="bg-yellow-50 text-yellow-700 hover:bg-yellow-50">
                Forecasting
              </Badge>
              <Badge variant="outline" className="bg-gray-100 hover:bg-gray-100">
                Weather
              </Badge>
              <span className="text-sm text-gray-500">ID: 192819210</span>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <span className="text-lg font-medium">2025</span>
              </div>
              <p className="text-sm text-gray-500">year</p>
            </div>

            <div className="space-y-1">
              <p className="text-lg font-medium">Indonesia</p>
              <p className="text-sm text-gray-500">Country</p>
            </div>

            <div className="space-y-1">
              <p className="text-lg font-medium">Asia</p>
              <p className="text-sm text-gray-500">Continent</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
