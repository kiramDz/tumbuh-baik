import { ColumnDef } from "@tanstack/react-table";
import { format } from "date-fns";
import { UserType } from "@/types/table-schema";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { updateUserRole } from "@/lib/fetch/files.fetch";
import { toast } from "sonner";
import { Shield, UserCog } from "lucide-react";

export const userColumns: ColumnDef<UserType>[] = [
	{
		accessorKey: "name",
		header: () => <div className="font-medium">User</div>,
		cell: ({ row }) => {
			const user = row.original;
			const initials =
				user.name
					?.split(" ")
					.map((n) => n[0])
					.join("")
					.toUpperCase()
					.slice(0, 2) || "U";

			return (
				<div className="flex items-center gap-3 min-w-[250px]">
					<Avatar className="h-10 w-10 shrink-0">
						<AvatarImage src={user.image} alt={user.name} />
						<AvatarFallback className="bg-primary/10 text-primary font-semibold text-sm">
							{initials}
						</AvatarFallback>
					</Avatar>
					<div className="flex flex-col gap-0.5 overflow-hidden">
						<span className="font-medium text-sm truncate">{user.name}</span>
						<span className="text-xs text-muted-foreground truncate">{user.email}</span>
					</div>
				</div>
			);
		},
		enableHiding: false,
	},
	{
		accessorKey: "role",
		header: () => <div className="font-medium">Role</div>,
		cell: ({ row }) => {
			const user = row.original;
			const userId = typeof user._id === "string" ? user._id : user._id.$oid;

			const handleRoleChange = async (newRole: "user" | "admin") => {
				if (newRole === user.role) return;

				try {
					await updateUserRole(userId, newRole);
					toast.success("Role updated", {
						description: `User role has been changed to ${newRole}`,
					});
					window.location.reload();
				} catch (error) {
					toast.error("Update failed", {
						description: "Unable to change user role. Please try again.",
					});
					console.error("Role update error:", error);
				}
			};

			return (
				<div className="w-[130px]">
					<Select value={user.role} onValueChange={(value: "user" | "admin") => handleRoleChange(value)}>
						<SelectTrigger className="h-9 w-full">
							<SelectValue />
						</SelectTrigger>
						<SelectContent>
							<SelectItem value="user">
								<div className="flex items-center gap-2">
									<UserCog className="h-4 w-4 text-muted-foreground" />
									<span>User</span>
								</div>
							</SelectItem>
							<SelectItem value="admin">
								<div className="flex items-center gap-2">
									<Shield className="h-4 w-4 text-muted-foreground" />
									<span>Admin</span>
								</div>
							</SelectItem>
						</SelectContent>
					</Select>
				</div>
			);
		},
	},
	{
		accessorKey: "createdAt",
		header: () => <div className="font-medium">Joined</div>,
		cell: ({ row }) => {
			const date = new Date(row.getValue("createdAt"));
			return (
				<div className="flex flex-col gap-1 min-w-[140px]">
					<span className="text-sm font-medium">{format(date, "MMM dd, yyyy")}</span>
					<span className="text-xs text-muted-foreground">{format(date, "HH:mm a")}</span>
				</div>
			);
		},
	},
	{
		accessorKey: "updatedAt",
		header: () => <div className="font-medium">Last Active</div>,
		cell: ({ row }) => {
			const date = new Date(row.getValue("updatedAt"));
			const now = new Date();
			const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));

			let timeAgo = "";
			let variant: "default" | "secondary" | "outline" = "secondary";

			if (diffInHours < 1) {
				timeAgo = "Just now";
				variant = "default";
			} else if (diffInHours < 24) {
				timeAgo = `${diffInHours}h ago`;
				variant = "default";
			} else {
				const diffInDays = Math.floor(diffInHours / 24);
				if (diffInDays === 1) {
					timeAgo = "Yesterday";
				} else if (diffInDays < 7) {
					timeAgo = `${diffInDays}d ago`;
				} else if (diffInDays < 30) {
					const weeks = Math.floor(diffInDays / 7);
					timeAgo = `${weeks}w ago`;
				} else {
					timeAgo = format(date, "MMM dd");
				}
				variant = "secondary";
			}

			return (
				<div className="flex flex-col gap-1 min-w-[140px]">
					<Badge variant={variant} className="w-fit text-xs font-normal">
						{timeAgo}
					</Badge>
					<span className="text-xs text-muted-foreground">{format(date, "MMM dd, yyyy")}</span>
				</div>
			);
		},
	},
];
