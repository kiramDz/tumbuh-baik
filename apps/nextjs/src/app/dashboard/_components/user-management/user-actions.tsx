"use client";

import type { UserType } from "@/types/table-schema";
import { usePathname } from "next/navigation";
import { type ComponentProps, useTransition } from "react";
import { toast } from "sonner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";

import Link from "next/link";
import { useSession } from "@/lib/better-auth/auth-client";
import { getErrorMessage } from "@/lib/handle-error";
import { updateUser } from "@/server/admin/user/actions";
import { cn } from "@/lib/utils";

type UserActionsProps = ComponentProps<typeof Button> & {
  user: UserType;
};

export const UserActions = ({ user, className, ...props }: UserActionsProps) => {
  const { data: session } = useSession();
  const pathname = usePathname();
  const [isUpdatePending, startUpdateTransition] = useTransition();
  const roles = ["admin", "user"] as const;

  if (user._id === session?.user.id) {
    return null;
  }

  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger asChild>
        <Button aria-label="Open menu" variant="secondary" size="sm" className={cn("data-[state=open]:bg-accent", className)} {...props} />
      </DropdownMenuTrigger>

      <DropdownMenuContent align="end" sideOffset={8}>
        {pathname !== `/admin/users/${user._id}` && (
          <DropdownMenuItem asChild>
            <Link href={`/admin/users/${user._id}`}>Edit</Link>
          </DropdownMenuItem>
        )}

        <DropdownMenuSub>
          <DropdownMenuSubTrigger>Role</DropdownMenuSubTrigger>

          <DropdownMenuSubContent>
            <DropdownMenuRadioGroup
              value={user.role}
              onValueChange={(value) => {
                startUpdateTransition(() => {
                  toast.promise(
                    async () =>
                      await updateUser({
                        id: user._id,
                        role: value as (typeof roles)[number],
                      }),
                    {
                      loading: "Updating...",
                      success: "Role successfully updated",
                      error: (err) => getErrorMessage(err),
                    }
                  );
                });
              }}
            >
              {roles.map((role) => (
                <DropdownMenuRadioItem key={role} value={role} className="capitalize" disabled={isUpdatePending}>
                  {role}
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
          </DropdownMenuSubContent>
        </DropdownMenuSub>

        <DropdownMenuSeparator />
      </DropdownMenuContent>
    </DropdownMenu>
  );
};
