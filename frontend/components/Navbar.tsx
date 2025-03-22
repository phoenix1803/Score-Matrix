"use client";

import React from 'react';
import { useAuth } from "../context/AuthContext";
import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Menu, X } from "lucide-react";
import { motion } from "framer-motion";

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { user, logout } = useAuth();
  const pathname = usePathname();
  const router = useRouter();
  const [navItems, setNavItems] = useState<{ name: string; href: string }[]>([]);
  const [animate, setAnimate] = useState(false);

  useEffect(() => {
    setNavItems(
      user
        ? [
          { name: "Home", href: "/dashboard" },
          { name: "Upload", href: "/upload" },
          { name: "Settings", href: "/settings" },
          { name: "Help", href: "/help" },
          { name: "Profile", href: "/profile" },
        ]
        : [{ name: "Login", href: "/login" }]
    );
  }, [user]);

  useEffect(() => {
    // Trigger logo animation on initial load
    setAnimate(true);

    // Reset and retrigger animation when hovered
    const interval = setInterval(() => {
      setAnimate(false);
      setTimeout(() => setAnimate(true), 50);
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const handleLogout = async () => {
    await logout();
    router.replace("/login");
  };

  return (
    <nav className="bg-white shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Animated Logo */}
          <div className="flex-shrink-0 flex items-center">
            <Link href="/" className="relative group">
              {/* Grid background */}
              <div className="absolute -inset-1 bg-gradient-to-r from-primary/20 to-primary/10 rounded-lg opacity-75 group-hover:opacity-100 transition-opacity duration-300">
                <div className="absolute inset-0 grid grid-cols-4 grid-rows-2 gap-px opacity-30 group-hover:opacity-50 transition-opacity duration-300">
                  {Array(8).fill(0).map((_, i) => (
                    <div key={i} className="bg-primary/40 rounded-sm" />
                  ))}
                </div>
              </div>

              {/* Main logo text */}
              <div className="relative px-4 py-2 text-2xl font-bold text-primary">
                <span className={`inline-block ${animate ? 'animate-fade-in' : 'opacity-0'} transition-all duration-500 delay-100`}>
                  Score
                </span>
                <span className={`inline-block ${animate ? 'animate-fade-in' : 'opacity-0'} transition-all duration-500 delay-300 relative`}>
                  Matrix
                  <span className="absolute -top-1 -right-1 h-1 w-1 bg-primary rounded-full animate-ping" />
                </span>
              </div>
            </Link>
          </div>

          <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
            {navItems.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${pathname === item.href
                    ? "border-primary text-primary"
                    : "border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700"
                  }`}
              >
                {item.name}
              </Link>
            ))}
            {user && (
              <button
                onClick={handleLogout}
                className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:border-gray-300 hover:text-gray-700"
              >
                Logout
              </button>
            )}
          </div>
          <div className="-mr-2 flex items-center sm:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary"
            >
              <span className="sr-only">Open main menu</span>
              {isOpen ? <X className="block h-6 w-6" /> : <Menu className="block h-6 w-6" />}
            </button>
          </div>
        </div>
      </div>

      <motion.div
        className={`sm:hidden ${isOpen ? "block" : "hidden"}`}
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        transition={{ duration: 0.2 }}
      >
        <div className="pt-2 pb-3 space-y-1">
          {navItems.map((item) => (
            <Link
              key={item.name}
              href={item.href}
              className={`block pl-3 pr-4 py-2 border-l-4 text-base font-medium ${pathname === item.href
                  ? "border-primary text-primary bg-primary-50"
                  : "border-transparent text-gray-500 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-700"
                }`}
            >
              {item.name}
            </Link>
          ))}
          {user && (
            <button
              onClick={handleLogout}
              className="block w-full text-left pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-gray-500 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-700"
            >
              Logout
            </button>
          )}
        </div>
      </motion.div>
    </nav>
  );
};

export default Navbar;