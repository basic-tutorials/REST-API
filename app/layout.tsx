import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Logic Gate Builder",
  description: "Create circuits using AND / OR / NOT gates",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
