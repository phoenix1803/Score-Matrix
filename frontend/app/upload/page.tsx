"use client";

import React, { useState } from "react";
import UploadReferenceFile from "../../components/UploadReferenceFile";
import UploadMultipleFiles from "../../components/UploadMultipleFiles";
import ThankYouPage from "../../components/ThankYouPage";

interface Parameters {
  [key: string]: string | number | boolean;
}

const UploadPage = () => {
  const [step, setStep] = useState(1);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [parameters, setParameters] = useState<Parameters>({});
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);

  const handleNextStep = () => {
    setStep(step + 1);
  };

  const handlePreviousStep = () => {
    setStep(step - 1);
  };

  const handleReferenceFileUpload = (file: File) => {
    setReferenceFile(file);
  };

  const handleParameterChange = (newParameters: Parameters) => {
    setParameters(newParameters);
  };

  const handleMultipleFilesUpload = (files: File[]) => {
    setUploadedFiles(files);
  };

  const handleSubmit = async () => {
    // Simulating backend processing
    await new Promise((resolve) => setTimeout(resolve, 2000));
    handleNextStep();
  };

  return (
    <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      {step === 1 && (
        <UploadReferenceFile
          onUpload={handleReferenceFileUpload}
          onParameterChange={handleParameterChange}
          onNext={handleNextStep}
        />
      )}
      {step === 2 && (
        <UploadMultipleFiles
          onUpload={handleMultipleFilesUpload}
          onSubmit={handleSubmit}
          onPrevious={handlePreviousStep}
        />
      )}
      {step === 3 && (
        <ThankYouPage
          referenceFile={referenceFile}
          parameters={parameters}
          uploadedFiles={uploadedFiles}
          onRestart={() => setStep(1)}
        />
      )}
    </div>
  );
};

export default UploadPage;