"use client";

import React from 'react';
import { useAuth } from "../context/AuthContext";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { user } = useAuth();
  const router = useRouter();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Wait for user to be initialized
    if (user !== null) {
      setLoading(false);
      if (!user) {
        router.replace("/");
      }
    }
  }, [user, router]);

  if (loading) {
    return <div>Loading...</div>; // Show a loading spinner or placeholder
  }

  if (!user) return null;

  return <>{children}</>;
};

export default ProtectedRoute;