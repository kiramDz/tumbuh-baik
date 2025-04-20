export default function MdxLayout({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string; // Make className optional
}) {
  // Combine className with default styles
  return <div className={`container-wrapper ${className || ""}`}>{children}</div>;
}
