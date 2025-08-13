"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { signOut, useSession } from "@/lib/better-auth/auth-client";
import { useRouter } from "next/navigation";
import Avatar from "boring-avatars";
// import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";

const HeaderProfile = () => {
  const session = useSession();
  const router = useRouter();
  const { isPending, data } = session;

  if (isPending) {
    return <Skeleton className="size-10 rounded-full" />;
  }

  // if (!data?.user) {
  //   return (
  //     <Button variant="outline" onClick={() => router.push("/sign-in")}>
  //       Login
  //     </Button>
  //   );
  // }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger>
        <Avatar name={data?.user.name || "User"} colors={["#0a0310", "#49007e", "#ff005b", "#ff7d10", "#ffb238"]} variant="sunset" size={40} />
      </DropdownMenuTrigger>
      <DropdownMenuContent className="px-2">
        <DropdownMenuLabel>Action</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          className="flex items-center justify-center gap-2 px-3 py-4"
          onClick={async () => {
            await signOut();
            router.push("/sign-in");
          }}
        >
          Log Out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default HeaderProfile;
