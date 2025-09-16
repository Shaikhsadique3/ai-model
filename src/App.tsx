import { useState, useEffect } from 'react';
import { FileText, Shield, TrendingUp, Users, AlertTriangle } from 'lucide-react';
import FileUpload from './components/FileUpload';
import DataPreview from './components/DataPreview';
import ProcessingStatus from './components/ProcessingStatus';
import ResultsSummary from './components/ResultsSummary';
import { ProcessingState, UploadedFile, PredictionResults } from './types';

function App() {
  const [processingState, setProcessingState] = useState<ProcessingState>('idle');
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [predictionResults, setPredictionResults] = useState<PredictionResults | null>(null);
  const [reportUrl, setReportUrl] = useState<string | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [processingProgress, setProcessingProgress] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let statusInterval: NodeJS.Timeout;

    if (fileId && processingState === 'processing') {
      statusInterval = setInterval(async () => {
        try {
          const response = await fetch(`/api/status/${fileId}`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const data = await response.json();

          setProcessingProgress(data.progress);
          if (data.status === 'completed') {
            setProcessingState('completed');
            setReportUrl(data.report_url);
            // Fetch prediction results once completed
            const resultsResponse = await fetch(`/api/predictions/${fileId}`); // Assuming a new endpoint for predictions
            if (!resultsResponse.ok) {
              throw new Error(`HTTP error! status: ${resultsResponse.status}`);
            }
            const resultsData = await resultsResponse.json();
            setPredictionResults(resultsData);
            clearInterval(statusInterval);
          } else if (data.status === 'failed') {
            setProcessingState('error');
            setErrorMessage(data.error || 'Unknown error during processing.');
            clearInterval(statusInterval);
          }
        } catch (error: any) {
          console.error('Error polling status:', error);
          setProcessingState('error');
          setErrorMessage(error.message || 'Failed to get processing status.');
          clearInterval(statusInterval);
        }
      }, 5000); // Poll every 5 seconds
    }

    return () => clearInterval(statusInterval);
  }, [fileId, processingState]);

  const handleFileUploaded = async (file: File) => {
    setProcessingState('uploading');
    setErrorMessage(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'File upload failed');
      }

      const data = await response.json();
      setFileId(data.file_id);
      setUploadedFile({ file_id: data.file_id, filename: file.name, total_rows: data.total_rows, columns: data.columns, preview_data: data.preview_data });
      setProcessingState('uploaded'); // Change to 'uploaded' to show DataPreview
    } catch (error: any) {
      console.error('Error uploading file:', error);
      setProcessingState('error');
      setErrorMessage(error.message || 'File upload failed.');
    }
  };

  const resetApp = () => {
    setProcessingState('idle');
    setUploadedFile(null);
    setPredictionResults(null);
    setReportUrl(null);
    setFileId(null);
    setProcessingProgress(0);
    setErrorMessage(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-primary-600 p-2 rounded-lg">
                <TrendingUp className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Churn Audit Service</h1>
                <p className="text-sm text-gray-600">AI-Powered Customer Retention Analytics</p>
              </div>
            </div>
            {processingState !== 'idle' && (
              <button
                onClick={resetApp}
                className="btn-secondary text-sm"
              >
                New Analysis
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {(processingState === 'idle' || processingState === 'uploading') && (
          <>
            {/* Hero Section */}
            <div className="text-center mb-12">
              <h2 className="text-4xl font-bold text-gray-900 mb-4">
                Upload Your Customer Data → Get Churn Insights
              </h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">
                Transform your customer data into actionable retention strategies with AI-powered churn prediction and professional reporting.
              </p>
            </div>

            {/* Benefits Section */}
            <div className="grid md:grid-cols-3 gap-8 mb-12">
              <div className="card text-center">
                <div className="bg-danger-50 w-12 h-12 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <AlertTriangle className="w-6 h-6 text-danger-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Identify Risky Customers</h3>
                <p className="text-gray-600">AI analyzes usage patterns, billing history, and engagement to predict churn risk with 85%+ accuracy.</p>
              </div>
              
              <div className="card text-center">
                <div className="bg-primary-50 w-12 h-12 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <FileText className="w-6 h-6 text-primary-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Retention Playbook</h3>
                <p className="text-gray-600">Get personalized email templates and intervention strategies for each risk segment.</p>
              </div>
              
              <div className="card text-center">
                <div className="bg-success-50 w-12 h-12 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <Users className="w-6 h-6 text-success-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Industry Benchmarks</h3>
                <p className="text-gray-600">Compare your churn rates against SaaS industry standards and identify improvement opportunities.</p>
              </div>
            </div>

            {/* Upload Section */}
            <div className="max-w-2xl mx-auto">
              <FileUpload onFileUploaded={handleFileUploaded} />
              
              {/* Data Safety */}
              <div className="mt-8 p-4 bg-gray-50 rounded-lg border border-gray-200">
                <div className="flex items-center space-x-2 mb-2">
                  <Shield className="w-5 h-5 text-green-600" />
                  <span className="font-semibold text-gray-900">Data Safety Guarantee</span>
                </div>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Files automatically deleted within 24 hours</li>
                  <li>• All customer IDs are anonymized in reports</li>
                  <li>• No data stored or shared with third parties</li>
                  <li>• Enterprise-grade security and encryption</li>
                </ul>
              </div>
            </div>
          </>
        )}

        {processingState === 'uploaded' && uploadedFile && (
          <DataPreview 
            file={uploadedFile} 
          />
        )}

        {(processingState === 'processing' || processingState === 'error' || processingState === 'uploading') && (
          <ProcessingStatus 
            state={processingState}
            progress={processingProgress}
            errorMessage={errorMessage}
            onRetry={resetApp}
          />
        )}

        {processingState === 'completed' && predictionResults && fileId && (
          <ResultsSummary 
            results={predictionResults}
            reportUrl={reportUrl}
            fileId={fileId}
          />
        )}
      </main>
    </div>
  );
}

export default App;