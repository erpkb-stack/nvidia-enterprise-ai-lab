import "./globals.css";
import "./lab.css";
import type { Metadata } from "next";
import Nav from "../components/Nav";

export const metadata: Metadata = {
  title: "NVIDIA Enterprise AI Solutions Architect Lab",
  description: "Enterprise RAG, inference performance engineering, solution architecture, GPU operations and OEM validation.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <Nav />
          <div className="content">{children}</div>
        </div>
      </body>
    </html>
  );
}
