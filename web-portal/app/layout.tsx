import type { Metadata } from "next";
import "./globals.css";
import { ReactQueryProvider } from "@/lib/providers/react-query-provider";

export const metadata: Metadata = {
  title: "VeriSafe Mapper Portal",
  description: "Mapping and review portal for VeriSafe contributors",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body>
        <ReactQueryProvider>
          {children}
        </ReactQueryProvider>
      </body>
    </html>
  );
}
