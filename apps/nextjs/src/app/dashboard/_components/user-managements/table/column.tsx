import { ColumnDef } from "@tanstack/react-table";
import { format } from "date-fns";
import { UserType } from "@/types/table-schema";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { updateUserRole } from "@/lib/fetch/files.fetch";
import { toast } from "sonner";

export const userColumns: ColumnDef<UserType>[] = [
  {
    accessorKey: "name",
    header: "Name",
    cell: ({ row }) => {
      const user = row.original;
      return (
        <div className="flex items-center gap-3">
          <Avatar className="h-8 w-8">
            <AvatarImage src={user.image} alt={user.name} />
            <AvatarFallback>{user.name?.charAt(0)?.toUpperCase() || "U"}</AvatarFallback>
          </Avatar>
          <span className="font-medium">{user.name}</span>
        </div>
      );
    },
  },
  {
    accessorKey: "email",
    header: "Email",
    cell: ({ row }) => (
      <div className="flex flex-col">
        <span>{row.getValue("email")}</span>
        <div className="flex items-center gap-1 mt-1">
          <Badge variant={row.original.emailVerified ? "default" : "secondary"} className="text-xs">
            {row.original.emailVerified ? "Verified" : "Unverified"}
          </Badge>
        </div>
      </div>
    ),
  },
  {
    accessorKey: "role",
    header: "Role",
    cell: ({ row }) => {
      const user = row.original;
      const userId = typeof user._id === "string" ? user._id : user._id.$oid;

      const handleRoleChange = async (newRole: "user" | "admin") => {
        try {
          await updateUserRole(userId, newRole);
          toast.success("User role updated successfully");
          // Refresh the page or update the table data
          window.location.reload();
        } catch (error) {
          toast.error("Failed to update user role");
          console.error("Role update error:", error);
        }
      };

      return (
        <Select value={user.role} onValueChange={(value: "user" | "admin") => handleRoleChange(value)}>
          <SelectTrigger className="w-24">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="user">User</SelectItem>
            <SelectItem value="admin">Admin</SelectItem>
          </SelectContent>
        </Select>
      );
    },
  },
  {
    accessorKey: "createdAt",
    header: "Created At",
    cell: ({ row }) => {
      const date = new Date(row.getValue("createdAt"));
      return format(date, "MMM dd, yyyy");
    },
  },
  {
    accessorKey: "updatedAt",
    header: "Last Updated",
    cell: ({ row }) => {
      const date = new Date(row.getValue("updatedAt"));
      return format(date, "MMM dd, yyyy HH:mm");
    },
  },
];
