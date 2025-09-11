import React from 'react';
import { Download, TrendingUp, AlertTriangle, Users, Target, Award } from 'lucide-react';
import { ResultsSummaryProps } from '../types';

const ResultsSummary: React.FC<ResultsSummaryProps> = ({ results, reportUrl }) => {
  const downloadReport = async () => {
    if (reportUrl) {
      try {
        const response = await fetch(reportUrl);
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'churn-audit-report.pdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } catch (error) {
        console.error('Error downloading report:', error);
      }
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'high': return 'text-red-600 bg-red-50 border-red-200';
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'low': return 'text-green-600 bg-green-50 border-green-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getBenchmarkIcon = () => {
    switch (results.benchmark_comparison.performance) {
      case 'above': return <Award className="w-5 h-5 text-green-600" />;
      case 'below': return <AlertTriangle className="w-5 h-5 text-red-600" />;
      default: return <Target className="w-5 h-5 text-yellow-600" />;
    }
  };

  const getBenchmarkMessage = () => {
    const { performance, your_churn_rate, industry_average } = results.benchmark_comparison;
    const diff = Math.abs(your_churn_rate - industry_average);
    
    switch (performance) {
      case 'above':
        return `Excellent! Your churn rate is ${diff.toFixed(1)}% below industry average.`;
      case 'below':
        return `Your churn rate is ${diff.toFixed(1)}% above industry average. Focus on retention.`;
      default:
        return `Your churn rate is close to industry average.`;
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Success Header */}
      <div className="text-center">
        <div className="flex justify-center mb-4">
          <div className="bg-green-100 p-3 rounded-full">
            <TrendingUp className="w-8 h-8 text-green-600" />
          </div>
        </div>
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Analysis Complete!</h2>
        <p className="text-xl text-gray-600">
          Your churn audit report is ready with actionable insights
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid md:grid-cols-4 gap-6">
        <div className="card text-center">
          <Users className="w-8 h-8 text-gray-600 mx-auto mb-3" />
          <div className="text-2xl font-bold text-gray-900">{results.total_customers.toLocaleString()}</div>
          <div className="text-sm text-gray-600">Total Customers</div>
        </div>
        
        <div className="card text-center">
          <AlertTriangle className="w-8 h-8 text-red-600 mx-auto mb-3" />
          <div className="text-2xl font-bold text-red-600">{results.risk_distribution.high_risk_count}</div>
          <div className="text-sm text-gray-600">High Risk ({results.risk_distribution.high_risk_percent.toFixed(1)}%)</div>
        </div>
        
        <div className="card text-center">
          <div className="w-8 h-8 bg-yellow-100 rounded-full flex items-center justify-center mx-auto mb-3">
            <div className="w-4 h-4 bg-yellow-500 rounded-full"></div>
          </div>
          <div className="text-2xl font-bold text-yellow-600">{results.risk_distribution.medium_risk_count}</div>
          <div className="text-sm text-gray-600">Medium Risk ({results.risk_distribution.medium_risk_percent.toFixed(1)}%)</div>
        </div>
        
        <div className="card text-center">
          <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
            <div className="w-4 h-4 bg-green-500 rounded-full"></div>
          </div>
          <div className="text-2xl font-bold text-green-600">{results.risk_distribution.low_risk_count}</div>
          <div className="text-sm text-gray-600">Low Risk ({results.risk_distribution.low_risk_percent.toFixed(1)}%)</div>
        </div>
      </div>

      {/* Risk Distribution Chart */}
      <div className="card">
        <h3 className="text-xl font-semibold text-gray-900 mb-6">Churn Risk Distribution</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-4 h-4 bg-red-500 rounded"></div>
              <span className="font-medium text-gray-700">High Risk</span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="w-64 bg-gray-200 rounded-full h-3">
                <div 
                  className="bg-red-500 h-3 rounded-full" 
                  style={{ width: `${results.risk_distribution.high_risk_percent}%` }}
                ></div>
              </div>
              <span className="text-sm font-medium text-gray-900 w-12">
                {results.risk_distribution.high_risk_percent.toFixed(1)}%
              </span>
            </div>
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-4 h-4 bg-yellow-500 rounded"></div>
              <span className="font-medium text-gray-700">Medium Risk</span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="w-64 bg-gray-200 rounded-full h-3">
                <div 
                  className="bg-yellow-500 h-3 rounded-full" 
                  style={{ width: `${results.risk_distribution.medium_risk_percent}%` }}
                ></div>
              </div>
              <span className="text-sm font-medium text-gray-900 w-12">
                {results.risk_distribution.medium_risk_percent.toFixed(1)}%
              </span>
            </div>
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-4 h-4 bg-green-500 rounded"></div>
              <span className="font-medium text-gray-700">Low Risk</span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="w-64 bg-gray-200 rounded-full h-3">
                <div 
                  className="bg-green-500 h-3 rounded-full" 
                  style={{ width: `${results.risk_distribution.low_risk_percent}%` }}
                ></div>
              </div>
              <span className="text-sm font-medium text-gray-900 w-12">
                {results.risk_distribution.low_risk_percent.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Top Churn Reasons & Benchmark */}
      <div className="grid md:grid-cols-2 gap-8">
        <div className="card">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">Top Churn Reasons</h3>
          <div className="space-y-4">
            {results.top_churn_reasons.slice(0, 5).map((reason, index) => (
              <div key={index} className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-6 h-6 bg-primary-100 rounded-full flex items-center justify-center text-xs font-medium text-primary-700">
                    {index + 1}
                  </div>
                  <span className="text-gray-700">{reason.reason}</span>
                </div>
                <div className="text-sm font-medium text-gray-900">
                  {reason.percentage.toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <h3 className="text-xl font-semibold text-gray-900 mb-6">Industry Benchmark</h3>
          <div className="space-y-4">
            <div className="flex items-center space-x-3 mb-4">
              {getBenchmarkIcon()}
              <span className="font-medium text-gray-900">Performance vs Industry</span>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Your Churn Rate</span>
                <span className="font-semibold text-gray-900">
                  {results.benchmark_comparison.your_churn_rate.toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Industry Average</span>
                <span className="font-semibold text-gray-900">
                  {results.benchmark_comparison.industry_average.toFixed(1)}%
                </span>
              </div>
            </div>
            
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-700">
                {getBenchmarkMessage()}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Download Report */}
      <div className="card text-center">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Complete Analysis Report</h3>
        <p className="text-gray-600 mb-6">
          Download your comprehensive PDF report with detailed insights, retention strategies, 
          and personalized email templates for each risk segment.
        </p>
        
        <button
          onClick={downloadReport}
          disabled={!reportUrl}
          className="btn-primary text-lg px-8 py-4"
        >
          <Download className="w-5 h-5 inline mr-2" />
          Download Full Report (PDF)
        </button>
        
        {!reportUrl && (
          <p className="text-sm text-gray-500 mt-2">
            Report is being generated...
          </p>
        )}
      </div>
    </div>
  );
};

export default ResultsSummary;