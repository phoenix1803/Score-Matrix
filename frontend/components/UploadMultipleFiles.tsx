"use client";

import React from 'react';
import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Upload, X } from "lucide-react";
import ProtectedRoute from "./ProtectedRoute";

interface UploadMultipleFilesProps {
  onUpload: (files: File[]) => void;
  onSubmit: (result: { message: string; downloadLink?: string; error?: string }) => void;
  onPrevious: () => void;
}

const UploadMultipleFiles: React.FC<UploadMultipleFilesProps> = ({ onUpload, onSubmit, onPrevious }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<{ message: string; downloadLink?: string; error?: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    if (!selectedFiles.length) return;
    setFiles([...files, ...selectedFiles]);
    onUpload([...files, ...selectedFiles]);
  };

  const removeFile = (index: number) => {
    const newFiles = [...files];
    newFiles.splice(index, 1);
    setFiles(newFiles);
    onUpload(newFiles);
  };

  const handleSubmit = async () => {
    if (!files.length) return;

    setIsProcessing(true);
    setResult(null);

    try {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append("files", file);
      });

      const response = await fetch("/api/upload-multiple", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Processing failed");
      }

      const data: { message: string; downloadLink?: string; error?: string } = await response.json();

      if (data.error) {
        alert(`Processing failed: ${data.error}`);
      } else {
        setResult(data);
        onSubmit(data);
      }
    } catch (error) {
      console.error("Upload failed:", error);
      alert(`Upload failed: ${(error as Error).message || "Unknown error"}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (result?.downloadLink) {
      window.open(result.downloadLink, "_blank");
    }
  };

  return (
    <ProtectedRoute>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-8">
        <h2 className="text-2xl font-bold mb-4">Upload Multiple Files</h2>
        <div
          className="border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors hover:border-gray-400"
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-2 text-sm text-gray-500">Click to upload or drag and drop multiple files</p>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            onChange={handleFileChange}
            multiple
            accept=".pdf,.doc,.docx"
            disabled={isProcessing}
          />
        </div>
        {files.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-xl font-semibold">Uploaded Files</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
              {files.map((file, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-gray-100 rounded">
                  <span className="truncate">{file.name}</span>
                  <button
                    onClick={() => removeFile(index)}
                    className="text-red-500 hover:text-red-700"
                    disabled={isProcessing}
                  >
                    <X size={20} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {result && (
          <div className="p-4 bg-green-100 rounded">
            <p className="font-semibold text-green-800">Processing complete!</p>
            <button
              onClick={handleDownload}
              className="mt-2 px-4 py-1 bg-green-600 text-white rounded hover:bg-green-700"
            >
              Download Result
            </button>
          </div>
        )}

        <div className="flex justify-between">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="px-6 py-2 rounded-md text-primary border border-primary font-semibold"
            onClick={onPrevious}
            disabled={isProcessing}
          >
            Previous
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`px-6 py-2 rounded-md text-white font-semibold ${files.length > 0 ? "bg-primary hover:bg-primary-dark" : "bg-gray-300 cursor-not-allowed"
              }`}
            onClick={handleSubmit}
            disabled={files.length === 0 || isProcessing}
          >
            {isProcessing ? "Processing..." : "Submit"}
          </motion.button>
        </div>
        {isProcessing && (
          <div className="flex justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary"></div>
          </div>
        )}
      </motion.div>
    </ProtectedRoute>
  );
};

export default UploadMultipleFiles;
