import "../styles/globals.css";
import React from 'react';
import { Inter } from "next/font/google";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import { AuthProvider } from "../context/AuthContext";
import type { ReactNode } from "react";
import ChatbotModal from "../components/ChatbotModal";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "ScoreMatrix",
  description: "Automated Answer Sheet Checking System",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="flex flex-col min-h-screen">
          <AuthProvider>
            <Navbar />
            <main className="flex-grow pb-8">{children}</main>
            <Footer />
            <ChatbotModal />
          </AuthProvider>
        </div>
      </body>
    </html>
  );
}
