import type { Metadata } from "next";

import { Footer } from "@/components/footer";
import { Navbar } from "@/components/navbar";

import "./globals.css";

export const metadata: Metadata = {
  title: "Octagon Intel",
  description: "Calibrated prefight betting intelligence for UFC event cards.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-ink antialiased">
        <div className="min-h-screen">
          <Navbar />
          <main>{children}</main>
          <Footer />
        </div>
      </body>
    </html>
  );
}
