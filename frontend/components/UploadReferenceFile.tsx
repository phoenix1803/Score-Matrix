"use client";

import React from 'react';
import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Upload } from "lucide-react";
import ProtectedRoute from "./ProtectedRoute";

interface UploadReferenceFileProps {
  onUpload: (file: File) => void;
  onParameterChange: (params: Record<string, number>) => void;
  onNext: () => void;
}

const UploadReferenceFile: React.FC<UploadReferenceFileProps> = ({ onUpload, onParameterChange, onNext }) => {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [parameters, setParameters] = useState<Record<string, number>>({
    Correctness: 40,
    Steps: 30,
    Accuracy: 30,
    Relevance: 0,
    Coherence: 0,
    Presentation: 0,
  });

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("parameters", JSON.stringify(parameters));

      const response = await fetch("/api/upload-reference", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload reference file");
      }

      const data = await response.json();
      console.log("Upload successful:", data);
      onUpload(selectedFile);
    } catch (error) {
      console.error("Error uploading reference file:", error);
      setError("Failed to upload file. Please try again.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleParameterChange = (param: string, value: number) => {
    const newParameters = { ...parameters, [param]: value };
    let total = Object.values(newParameters).reduce((sum, val) => sum + val, 0);

    // Auto-adjust other parameters to ensure sum is 100%
    if (total !== 100) {
      const adjustableKeys = Object.keys(newParameters).filter((key) => key !== param && newParameters[key] > 0);
      const adjustment = (100 - total) / adjustableKeys.length;
      adjustableKeys.forEach((key) => {
        newParameters[key] = Math.max(0, Math.round((newParameters[key] + adjustment) * 10) / 10);
      });
      total = Object.values(newParameters).reduce((sum, val) => sum + val, 0);
    }

    setParameters(newParameters);
    onParameterChange(newParameters);
  };

  const isNextEnabled =
    file && Math.abs(Object.values(parameters).reduce((sum, val) => sum + val, 0) - 100) < 0.1;

  return (
    <ProtectedRoute>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-8">
        <h2 className="text-2xl font-bold mb-4">Upload Question Paper</h2>

        {/* File Upload Section */}
        <div
          className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${file ? "border-primary" : "border-gray-300 hover:border-gray-400"
            }`}
          onClick={() => fileInputRef.current?.click()}
        >
          {isUploading ? (
            <div className="flex flex-col items-center">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
              <p className="mt-2 text-sm text-gray-500">Uploading...</p>
            </div>
          ) : file ? (
            <div>
              <Upload className="mx-auto h-12 w-12 text-primary" />
              <p className="mt-2 text-sm text-gray-500">{file.name}</p>
            </div>
          ) : (
            <div>
              <Upload className="mx-auto h-12 w-12 text-gray-400" />
              <p className="mt-2 text-sm text-gray-500">Click to upload or drag and drop</p>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            onChange={handleFileChange}
            accept=".pdf,.doc,.docx"
            disabled={isUploading}
          />
        </div>

        {/* Error Message */}
        {error && <p className="text-red-500 text-sm">{error}</p>}

        {/* Parameter Selection */}
        {file && (
          <div className="space-y-4">
            <h3 className="text-xl font-semibold">Parameter Selection</h3>
            <p className="text-sm text-gray-500">
              Total: {Object.values(parameters).reduce((sum, val) => sum + val, 0).toFixed(1)}%
            </p>
            {Object.entries(parameters).map(([param, value]) => (
              <div key={param} className="flex items-center space-x-4">
                <label className="w-32">{param}</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="0.1"
                  value={value}
                  onChange={(e) => handleParameterChange(param, Number(e.target.value))}
                  className="flex-grow"
                />
              </div>
            ))}
          </div>
        )}

        {/* Next Button */}
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className={`px-6 py-2 rounded-md text-white font-semibold ${isNextEnabled ? "bg-primary hover:bg-primary-dark" : "bg-gray-300 cursor-not-allowed"
            }`}
          onClick={onNext}
          disabled={!isNextEnabled || isUploading}
        >
          Next
        </motion.button>
      </motion.div>
    </ProtectedRoute>
  );
};

export default UploadReferenceFile;
