import React, { useState, useRef } from 'react';
import { Upload, FileText, Download } from 'lucide-react';
import { FileUploadProps } from '../types';

const FileUpload: React.FC<FileUploadProps> = ({ onFileUploaded }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Please upload a CSV file');
      return;
    }

    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      setError('File size must be less than 10MB');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/upload-csv', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const result = await response.json();
      onFileUploaded(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const downloadSampleCSV = async () => {
    try {
      const response = await fetch('/api/sample-csv');
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'sample_customer_data.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (error) {
      console.error('Error downloading sample CSV:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Zone */}
      <div
        className={`upload-zone ${isDragOver ? 'dragover' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileSelect}
          className="hidden"
        />
        
        <div className="space-y-4">
          <div className="flex justify-center">
            <div className="bg-primary-100 p-4 rounded-full">
              <Upload className="w-8 h-8 text-primary-600" />
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              {isUploading ? 'Uploading...' : 'Drop your CSV file here'}
            </h3>
            <p className="text-gray-600 mb-4">
              or <span className="text-primary-600 font-medium cursor-pointer hover:text-primary-700">browse to choose a file</span>
            </p>
            <p className="text-sm text-gray-500">
              Supports CSV files up to 10MB with customer data
            </p>
          </div>
          
          {isUploading && (
            <div className="flex justify-center">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      {/* Sample CSV Download */}
      <div className="text-center">
        <button
          onClick={downloadSampleCSV}
          className="inline-flex items-center space-x-2 text-primary-600 hover:text-primary-700 font-medium text-sm"
        >
          <Download className="w-4 h-4" />
          <span>Download Sample CSV</span>
        </button>
        <p className="text-xs text-gray-500 mt-1">
          See the required format and column structure
        </p>
      </div>

      {/* Required Columns Info */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <FileText className="w-5 h-5 text-blue-600 mt-0.5" />
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Required CSV Columns:</h4>
            <div className="grid grid-cols-2 gap-2 text-sm text-blue-800">
              <div>• user_id</div>
              <div>• signup_date</div>
              <div>• last_login_timestamp</div>
              <div>• billing_status</div>
              <div>• plan_name</div>
              <div>• monthly_revenue</div>
            </div>
            <p className="text-xs text-blue-700 mt-2">
              Optional: support_tickets_opened, email_opens_last30days, last_payment_status
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FileUpload;