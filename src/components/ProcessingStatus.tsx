import React from 'react';
import { Loader2, AlertCircle, RefreshCw } from 'lucide-react';
import { ProcessingStatusProps } from '../types';

const ProcessingStatus: React.FC<ProcessingStatusProps> = ({ state, errorMessage, onRetry }) => {
  if (state === 'processing') {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="card text-center">
          <div className="space-y-6">
            <div className="flex justify-center">
              <div className="bg-primary-100 p-4 rounded-full">
                <Loader2 className="w-12 h-12 text-primary-600 animate-spin" />
              </div>
            </div>
            
            <div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                Analyzing Your Customer Data
              </h2>
              <p className="text-gray-600 mb-6">
                Our AI is processing your data to identify churn patterns and generate insights...
              </p>
            </div>

            {/* Progress Steps */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3 text-left">
                <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                </div>
                <span className="text-gray-700">Data validation and cleaning</span>
              </div>
              
              <div className="flex items-center space-x-3 text-left">
                <div className="w-6 h-6 bg-primary-500 rounded-full flex items-center justify-center animate-pulse">
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                </div>
                <span className="text-gray-700">Running churn prediction model</span>
              </div>
              
              <div className="flex items-center space-x-3 text-left">
                <div className="w-6 h-6 bg-gray-300 rounded-full flex items-center justify-center">
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                </div>
                <span className="text-gray-500">Generating insights and recommendations</span>
              </div>
              
              <div className="flex items-center space-x-3 text-left">
                <div className="w-6 h-6 bg-gray-300 rounded-full flex items-center justify-center">
                  <div className="w-2 h-2 bg-white rounded-full"></div>
                </div>
                <span className="text-gray-500">Creating PDF report</span>
              </div>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-sm text-blue-800">
                <strong>Did you know?</strong> Our AI model analyzes over 20 behavioral patterns 
                to predict churn with 85%+ accuracy, helping you retain more customers.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (state === 'error') {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="card text-center">
          <div className="space-y-6">
            <div className="flex justify-center">
              <div className="bg-red-100 p-4 rounded-full">
                <AlertCircle className="w-12 h-12 text-red-600" />
              </div>
            </div>
            
            <div>
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">
                Processing Failed
              </h2>
              {errorMessage && (
                <p className="text-red-600 mb-4 font-medium">Error: {errorMessage}</p>
              )}
              <p className="text-gray-600 mb-6">
                We encountered an error while analyzing your data. This could be due to:
              </p>
              
              <ul className="text-left text-gray-600 space-y-2 mb-6">
                <li>• Missing required columns in your CSV file</li>
                <li>• Invalid data format or corrupted file</li>
                <li>• Temporary server issue</li>
              </ul>
            </div>

            <div className="flex justify-center space-x-4">
              <button
                onClick={onRetry}
                className="btn-primary"
              >
                <RefreshCw className="w-4 h-4 inline mr-2" />
                Try Again
              </button>
              <button
                onClick={() => window.location.reload()}
                className="btn-secondary"
              >
                Upload New File
              </button>
            </div>

            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <p className="text-sm text-yellow-800">
                <strong>Need help?</strong> Make sure your CSV includes the required columns: 
                user_id, signup_date, last_login_timestamp, billing_status, plan_name, monthly_revenue
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

export default ProcessingStatus;