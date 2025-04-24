import { Children } from "@/props/types";

const Layout = ({ children }: Children) => {
  return (
    <main className="flex items-center justify-center w-full h-screen">
      <div className="flex-1 w-full h-full flex items-center justify-center ">{children}</div>
    </main>
  );
};

export default Layout;
