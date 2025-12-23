"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { signOut, useSession } from "@/lib/better-auth/auth-client";
import { useRouter } from "next/navigation";
import Avatar from "boring-avatars";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { LayoutDashboard, LogOut, ChevronDown } from "lucide-react";

const HeaderProfile = () => {
  const session = useSession();
  const router = useRouter();
  const { isPending, data } = session;

  const isAdmin = data?.user?.role === "admin";
  const userName = data?.user?.name || "User";
  const userEmail = data?.user?.email || "";

  if (isPending) {
    return <Skeleton className="h-9 w-9 rounded-full" />;
  }

  if (!data?.user) {
    return (
      <Button size="sm" onClick={() => router.push("/sign-in")}>
        Login
      </Button>
    );
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button className="flex items-center gap-2 rounded-full p-0.5 pr-2 hover:bg-muted/60 transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring">
          <Avatar
            name={userName}
            colors={["#0a0310", "#49007e", "#ff005b", "#ff7d10", "#ffb238"]}
            variant="sunset"
            size={32}
          />
          <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
        </button>
      </DropdownMenuTrigger>

      <DropdownMenuContent align="end" className="w-56">
        <div className="px-3 py-2.5">
          <p className="text-sm font-medium truncate">{userName}</p>
          <p className="text-xs text-muted-foreground truncate">{userEmail}</p>
        </div>

        <DropdownMenuSeparator />

        {isAdmin && (
          <>
            <DropdownMenuItem
              onClick={() => router.push("/dashboard")}
              className="gap-2 py-2"
            >
              <LayoutDashboard className="h-4 w-4" />
              <span>Dashboard</span>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
          </>
        )}

        <DropdownMenuItem
          onClick={async () => {
            await signOut();
            router.push("/sign-in");
          }}
          className="gap-2 py-2 text-destructive focus:text-destructive"
        >
          <LogOut className="h-4 w-4" />
          <span>Logout</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default HeaderProfile;