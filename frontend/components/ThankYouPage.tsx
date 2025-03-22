"use client";

import React from "react";
import { motion } from "framer-motion";
import { Download, Share2, RefreshCw } from "lucide-react";
import ProtectedRoute from "./ProtectedRoute";
import { useRouter } from "next/navigation";
import { useState } from "react";

// Define a type for the parameters object
interface Parameters {
  [key: string]: string | number | boolean; // Adjust based on your actual parameter structure
}

interface ThankYouPageProps {
  referenceFile: File | null;
  parameters: Parameters; // Use the defined type
  uploadedFiles: File[];
  onRestart: () => void;
}

const ThankYouPage = ({
  referenceFile,
  parameters,
  uploadedFiles,
  onRestart,
}: ThankYouPageProps) => {
  const router = useRouter();
  void referenceFile;
  void parameters;
  void uploadedFiles;
  void onRestart;
  const [reportData, setReportData] = useState<unknown>(null); // Use `unknown` for dynamic data
  const [isDownloaded, setIsDownloaded] = useState(false); // Track if the download button has been clicked

  const handleDownload = async () => {
    try {
      const response = await fetch("/api/convert-json-to-excel", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ filePath: "path/to/your/report.json" }), // Replace with actual JSON file path
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "report.xlsx";
        a.click();
        window.URL.revokeObjectURL(url);

        // Disable the button after successful download
        setIsDownloaded(true);
      } else {
        console.error("Failed to convert JSON to Excel");
      }
    } catch (error) {
      console.error("Error downloading report:", error);
    }
  };

  const handleShare = async () => {
    try {
      const response = await fetch("/api/share-report", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ filePath: "path/to/your/report.json" }), // Replace with actual JSON file path
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Shareable link:", data.shareableLink);
        alert(`Report shared successfully! Link: ${data.shareableLink}`);
      } else {
        console.error("Failed to share report");
      }
    } catch (error) {
      console.error("Error sharing report:", error);
    }
  };

  // Handle view report: Fetch and display JSON data
  const handleViewReport = async () => {
    try {
      const response = await fetch("/api/get-report-data", {
        method: "GET",
      });

      if (response.ok) {
        const data = await response.json();
        setReportData(data);
      } else {
        console.error("Failed to fetch report data");
      }
    } catch (error) {
      console.error("Error viewing report:", error);
    }
  };

  // Redirect to dashboard
  const handleDashboardRedirect = () => {
    router.push("/dashboard");
  };

  // Type guard to check if reportData is an object
  const isReportData = (data: unknown): data is Record<string, unknown> => {
    return typeof data === "object" && data !== null;
  };

  return (
    <ProtectedRoute>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="space-y-8 text-center"
      >
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 260, damping: 20 }}
          className="w-24 h-24 bg-green-500 rounded-full flex items-center justify-center mx-auto"
        >
          <svg
            className="w-12 h-12 text-white"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M5 13l4 4L19 7"
            ></path>
          </svg>
        </motion.div>
        <h2 className="text-3xl font-bold">Thank You!</h2>
        <p className="text-xl">Your files have been successfully processed.</p>
        <div className="flex justify-center space-x-4">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="px-6 py-2 rounded-md bg-primary text-white font-semibold flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleDownload}
            disabled={isDownloaded}
          >
            <Download className="mr-2" size={20} />
            {isDownloaded ? "Downloaded" : "Download Report"}
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="px-6 py-2 rounded-md bg-primary text-white font-semibold flex items-center"
            onClick={handleShare}
          >
            <Share2 className="mr-2" size={20} />
            Share Report
          </motion.button>
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="px-6 py-2 rounded-md border border-primary text-primary font-semibold"
          onClick={handleViewReport}
        >
          View Report
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="px-6 py-2 rounded-md bg-tertiary text-white font-semibold flex items-center justify-center mx-auto"
          onClick={handleDashboardRedirect}
        >
          <RefreshCw className="mr-2" size={20} />
          Go to Dashboard
        </motion.button>

        {/* Display report data */}
        {isReportData(reportData) && (
          <div className="mt-8 p-4 bg-gray-100 rounded-lg text-left">
            <h3 className="text-xl font-bold mb-4">Report Data</h3>
            <pre>{JSON.stringify(reportData, null, 2)}</pre>
          </div>
        )}
      </motion.div>
    </ProtectedRoute>
  );
};

export default ThankYouPage;