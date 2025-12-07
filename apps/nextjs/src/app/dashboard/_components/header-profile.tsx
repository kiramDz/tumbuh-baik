"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { signOut, useSession } from "@/lib/better-auth/auth-client";
import { useRouter } from "next/navigation";
import Avatar from "boring-avatars";
import { Button } from "@/components/ui/button";
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuGroup,
  DropdownMenuItem, 
  DropdownMenuLabel, 
  DropdownMenuSeparator, 
  DropdownMenuTrigger 
} from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";
import { LogOut, LogIn, Settings, Shield, LayoutDashboard, ChevronRight } from "lucide-react";
import { memo, useState } from "react";

// Memoized Avatar component untuk mencegah re-render
const MemoizedAvatar = memo(({ name, size }: { name: string; size: number }) => (
  <Avatar 
    name={name} 
    colors={["#0a0310", "#49007e", "#ff005b", "#ff7d10", "#ffb238"]} 
    variant="sunset" 
    size={size} 
  />
));
MemoizedAvatar.displayName = "MemoizedAvatar";

const HeaderProfile = () => {
  const session = useSession();
  const router = useRouter();
  const { isPending, data } = session;
  const [loading, setLoading] = useState(false);

  const handleSignIn = async () => {
    setLoading(true);
    try {
      await router.push("/sign-in");
    } finally {
      setLoading(false);
    }
  };

  const isAdmin = data?.user?.role === "admin";

  const handleDashboard = () => {
    if (isAdmin) {
      router.push("/dashboard");
    } else {
      router.push("/user-dashboard");
    }
  };

  const handleSettings = () => {
    router.push("/settings");
  };

  const handleSignOut = async () => {
    setLoading(true);
    try {
      await signOut();
      router.push("/");
    } finally {
      setLoading(false);
    }
  };

  if (isPending || loading) {
    return <Skeleton className="h-10 w-10 rounded-full" />;
  }

  if (!data?.user) {
    return (
      <Button 
        variant="default" 
        size="sm"
        onClick={handleSignIn}
      >
        <LogIn className="mr-2 h-4 w-4" />
        Sign In
      </Button>
    );
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" className="p-0 h-10 w-10 rounded-full overflow-hidden hover:ring-2 hover:ring-primary/20 transition-all">
          <MemoizedAvatar name={data.user.name} size={40} />
        </Button>
      </DropdownMenuTrigger>
      
      <DropdownMenuContent className="w-64" align="end" sideOffset={8}>
        <DropdownMenuLabel className="font-normal">
          <div className="flex items-center gap-3 py-2">
            <div className="h-10 w-10 rounded-full overflow-hidden flex-shrink-0">
              <MemoizedAvatar name={data.user.name} size={40} />
            </div>
            <div className="flex flex-col space-y-1 flex-1 min-w-0">
              <p className="text-sm font-semibold leading-none truncate">{data.user.name}</p>
              <p className="text-xs text-muted-foreground leading-none truncate">{data.user.email}</p>
            </div>
          </div>
        </DropdownMenuLabel>
        
        <DropdownMenuSeparator />
        
        {isAdmin && (
          <>
            <div className="px-2 py-1.5">
              <Badge variant="secondary" className="w-full justify-center">
                <Shield className="h-3 w-3 mr-1" />
                Administrator
              </Badge>
            </div>
            <DropdownMenuSeparator />
          </>
        )}
        
        <DropdownMenuGroup>
          <DropdownMenuItem onClick={handleDashboard} className="cursor-pointer py-2.5">
            <div className="flex items-center justify-between w-full">
              <div className="flex items-center gap-2">
                {isAdmin ? (
                  <div className="h-8 w-8 rounded-md bg-purple-100 dark:bg-purple-900/20 flex items-center justify-center">
                    <Shield className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                  </div>
                ) : (
                  <div className="h-8 w-8 rounded-md bg-blue-100 dark:bg-blue-900/20 flex items-center justify-center">
                    <LayoutDashboard className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                  </div>
                )}
                <span className="font-medium">{isAdmin ? "Admin Dashboard" : "Dashboard User"}</span>
              </div>
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            </div>
          </DropdownMenuItem>
          
          <DropdownMenuItem onClick={handleSettings} className="cursor-pointer py-2.5">
            <div className="flex items-center justify-between w-full">
              <div className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-md bg-gray-100 dark:bg-gray-900/20 flex items-center justify-center">
                  <Settings className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                </div>
                <span>Settings</span>
              </div>
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            </div>
          </DropdownMenuItem>
        </DropdownMenuGroup>
        
        <DropdownMenuSeparator />
        
        <DropdownMenuItem 
          onClick={handleSignOut} 
          className="cursor-pointer py-2.5 text-red-600 focus:text-red-600 dark:text-red-400 dark:focus:text-red-400 focus:bg-red-50 dark:focus:bg-red-900/10"
        >
          <div className="flex items-center gap-2 w-full">
            <div className="h-8 w-8 rounded-md bg-red-100 dark:bg-red-900/20 flex items-center justify-center">
              <LogOut className="h-4 w-4 text-red-600 dark:text-red-400" />
            </div>
            <span className="font-medium">Log out</span>
          </div>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default HeaderProfile;
