'use client';

import { useState, useCallback } from 'react';
import Link from 'next/link';

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      validateAndSetFile(event.target.files[0]);
    }
  };

  const onDragEnter = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  }, []);

  const validateAndSetFile = useCallback((file: File) => {
    if (!file.name.endsWith('.csv')) {
      alert('Please upload a CSV file only');
      return;
    }
    
    if (file.size > MAX_FILE_SIZE) {
      alert('File size exceeds 5MB limit');
      return;
    }
    
    setSelectedFile(file);
  }, [MAX_FILE_SIZE]);

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  }, [validateAndSetFile]);

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a file first!');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/upload-csv', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        alert(`File uploaded successfully!\nFile Path: ${result.file_path}\nRows: ${result.rows}\nColumns: ${result.columns.join(', ')}`);
        setSelectedFile(null); // Clear selected file after successful upload
      } else {
        const errorData = await response.json();
        alert(`File upload failed: ${errorData.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file.');
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <header className="w-full py-12 bg-white shadow-sm">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Churn Audit Service</h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Upload your customer data to generate a comprehensive churn audit report.
            Understand why customers leave and how to retain them.
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow container mx-auto px-4 py-16 flex flex-col items-center justify-center">
        <section className="bg-white p-8 rounded-lg shadow-lg w-full max-w-md">
          <h2 className="text-2xl font-semibold text-gray-800 mb-6 text-center">Upload Your CSV</h2>
          
          <div 
            className={`border-2 border-dashed rounded-lg p-8 mb-4 text-center cursor-pointer transition-colors ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}
            onDragEnter={onDragEnter}
            onDragLeave={onDragLeave}
            onDragOver={(e) => e.preventDefault()}
            onDrop={onDrop}
            onClick={() => document.getElementById('file-upload')?.click()}
          >
            <div className="flex flex-col items-center justify-center">
              <svg className="w-12 h-12 text-gray-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="text-sm text-gray-600 mb-1">
                {isDragActive ? 'Drop your CSV file here' : 'Drag & drop your CSV file here'}
              </p>
              <p className="text-xs text-gray-500 mb-3">or</p>
              <button 
                type="button"
                className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Browse Files
              </button>
              <input
                id="file-upload"
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
              />
              <p className="text-xs text-gray-500 mt-3">CSV files only (max 5MB)</p>
            </div>
          </div>

          {selectedFile && (
            <div className="bg-gray-50 p-3 rounded-md flex items-center justify-between">
              <div className="flex items-center">
                <svg className="w-5 h-5 text-gray-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span className="text-sm font-medium text-gray-700">{selectedFile.name}</span>
              </div>
              <button 
                type="button" 
                className="text-sm text-blue-600 hover:text-blue-800"
                onClick={() => setSelectedFile(null)}
              >
                Replace
              </button>
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={!selectedFile}
            className="w-full mt-4 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50"
          >
            Generate Report
          </button>

          {/* Download Sample CSV Template */}
          <div className="mt-4 text-center">
            <a
              href="#"
              className="text-blue-600 hover:underline text-sm relative group"
              onClick={(e) => e.preventDefault()} // Prevent actual navigation for now
            >
              Download Sample CSV Template
              <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden w-max rounded bg-gray-800 p-2 text-xs text-white opacity-0 group-hover:opacity-100 group-hover:block transition-opacity duration-300">
                Required columns: CustomerID, Churn, MonthlyRevenue, TotalPurchases, LastPurchaseDate
              </span>
            </a>
          </div>
        </section>
      </main>

        {/* Data Safety & Trust Section */}
        <section className="mt-12 bg-white p-8 rounded-lg shadow-lg w-full max-w-md text-center">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">Your Data is Safe with Us</h2>
          <p className="text-gray-600 mb-4">
            We prioritize the security and privacy of your sensitive customer data. All uploads are encrypted,
            processed securely, and never shared with third parties. We are fully GDPR compliant.
          </p>
          <Link href="/privacy" className="text-blue-600 hover:underline">
            Read our Privacy Policy
          </Link>
        </section>

        {/* Footer */}
        <footer className="w-full py-8 bg-gray-800 text-white text-center">
          <div className="container mx-auto px-4">
            <p>&copy; 2024 Churn Audit Service. All rights reserved.</p>
            <div className="mt-4">
              <Link href="/privacy" className="text-gray-400 hover:text-white mx-2">Privacy Policy</Link>
              <Link href="/terms" className="text-gray-400 hover:text-white mx-2">Terms of Service</Link>
            </div>
          </div>
        </footer>
      </div>
    );
}

          <div className="flex flex-col min-h-screen">
            <main className="flex-1">
              {/* Header Section */}
              <section className="w-full py-12 md:py-24 lg:py-32">
                <div className="container px-4 md:px-6">
                  <div className="flex flex-col items-center space-y-4 text-center">
                    <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none">
                      Churn Audit Service
                    </h1>
                    <p className="max-w-[600px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                      Upload your customer data to generate a comprehensive churn analysis report.
                    </p>
                  </div>
                </div>
              </section>
          
              {/* Upload Section - Made responsive */}
              <section className="w-full py-12 md:py-24 lg:py-32 border-t">
                <div className="container px-4 md:px-6">
                  <div className="grid gap-6 px-4 md:px-6">
                    <div className="flex flex-col items-center space-y-4 text-center">
                      <div className="space-y-2 w-full max-w-md">
                        {/* File upload component with responsive sizing */}
                        <div className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-6 w-full">
                          {/* ... upload component content ... */}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </section>
          
              {/* Benefits Section - Already responsive */}
              <section className="w-full py-12 md:py-24 lg:py-32 bg-gray-100 dark:bg-gray-800">
                {/* ... existing benefits section ... */}
              </section>
          
              {/* Data Safety Section - Made responsive */}
              <section className="w-full py-12 md:py-24 lg:py-32">
                <div className="container px-4 md:px-6">
                  <div className="flex flex-col items-center space-y-4 text-center">
                    <div className="space-y-2 max-w-[800px]">
                      {/* ... safety section content ... */}
                    </div>
                  </div>
                </div>
              </section>
            </main>
          </div>
          {/* Benefits Section */}
          <section className="w-full py-12 md:py-24 lg:py-32 bg-gray-100 dark:bg-gray-800">
            <div className="container px-4 md:px-6">
              <div className="flex flex-col items-center justify-center space-y-4 text-center">
                <div className="space-y-2">
                  <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">Why Churn Audit Service?</h2>
                  <p className="max-w-[900px] text-gray-500 md:text-xl/relaxed lg:text-base/relaxed xl:text-xl/relaxed dark:text-gray-400">
                    Unlock the power of your customer data to understand and reduce churn.
                  </p>
                </div>
              </div>
              <div className="mx-auto grid max-w-5xl items-center gap-6 py-12 lg:grid-cols-3 lg:gap-12">
                <div className="flex flex-col items-center space-y-2">
                  <svg
                    className="h-12 w-12 text-gray-900 dark:text-gray-50"
                    fill="none"
                    height="24"
                    stroke="currentColor"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                    width="24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path d="M2 12s3-9 10-9 10 9 10 9-3 9-10 9-10-9-10-9Z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                  <h3 className="text-xl font-bold">Deep Insights</h3>
                  <p className="text-gray-500 dark:text-gray-400 text-center">
                    Gain a comprehensive understanding of why your customers are churning.
                  </p>
                </div>
                <div className="flex flex-col items-center space-y-2">
                  <svg
                    className="h-12 w-12 text-gray-900 dark:text-gray-50"
                    fill="none"
                    height="24"
                    stroke="currentColor"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                    width="24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                  </svg>
                  <h3 className="text-xl font-bold">Data Security</h3>
                  <p className="text-gray-500 dark:text-gray-400 text-center">
                    Your data is safe with us. We prioritize privacy and compliance.
                  </p>
                </div>
                <div className="flex flex-col items-center space-y-2">
                  <svg
                    className="h-12 w-12 text-gray-900 dark:text-gray-50"
                    fill="none"
                    height="24"
                    stroke="currentColor"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                    width="24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path d="M22 11V7a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-4" />
                    <path d="M22 11h-6a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-4a2 2 0 0 0-2-2z" />
                    <path d="M14 2v4" />
                    <path d="M20 2v4" />
                    <path d="M17 11V2h-2" />
                  </svg>
                  <h3 className="text-xl font-bold">Actionable Reports</h3>
                  <p className="text-gray-500 dark:text-gray-400 text-center">
                    Receive clear, actionable reports to guide your retention strategies.
                  </p>
                </div>
              </div>
            </div>
          </section>
