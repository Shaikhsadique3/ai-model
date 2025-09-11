import React, { useState, useEffect } from 'react';
import { FileText, Users, Eye, Play } from 'lucide-react';
import { DataPreviewProps } from '../types';

const DataPreview: React.FC<DataPreviewProps> = ({ file, onGenerateReport }) => {
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPreview = async () => {
      try {
        const response = await fetch(`/api/preview?file_id=${file.file_id}`);
        if (response.ok) {
          const data = await response.json();
          setPreviewData(data.preview_data || []);
        }
      } catch (error) {
        console.error('Error fetching preview:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchPreview();
  }, [file.file_id]);

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* File Info */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="bg-green-100 p-2 rounded-lg">
              <FileText className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">File Uploaded Successfully</h2>
              <p className="text-gray-600">{file.filename}</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-gray-900">{file.total_rows.toLocaleString()}</div>
            <div className="text-sm text-gray-600">customers</div>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-6 mb-6">
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <Users className="w-6 h-6 text-gray-600 mx-auto mb-2" />
            <div className="text-lg font-semibold text-gray-900">{file.total_rows.toLocaleString()}</div>
            <div className="text-sm text-gray-600">Total Customers</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <Eye className="w-6 h-6 text-gray-600 mx-auto mb-2" />
            <div className="text-lg font-semibold text-gray-900">{file.columns.length}</div>
            <div className="text-sm text-gray-600">Data Columns</div>
          </div>
          <div className="text-center p-4 bg-primary-50 rounded-lg">
            <Play className="w-6 h-6 text-primary-600 mx-auto mb-2" />
            <div className="text-lg font-semibold text-primary-900">Ready</div>
            <div className="text-sm text-primary-700">For Analysis</div>
          </div>
        </div>

        {/* Generate Report Button */}
        <div className="text-center">
          <button
            onClick={onGenerateReport}
            className="btn-primary text-lg px-8 py-4"
          >
            <Play className="w-5 h-5 inline mr-2" />
            Generate Churn Report
          </button>
          <p className="text-sm text-gray-600 mt-2">
            Analysis typically takes 30-60 seconds
          </p>
        </div>
      </div>

      {/* Data Preview */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Preview (First 5 Rows)</h3>
        
        {loading ? (
          <div className="flex justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
          </div>
        ) : previewData.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {file.columns.map((column) => (
                    <th
                      key={column}
                      className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {previewData.slice(0, 5).map((row, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    {file.columns.map((column) => (
                      <td key={column} className="px-4 py-3 text-sm text-gray-900 whitespace-nowrap">
                        {row[column] || '-'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            No preview data available
          </div>
        )}
      </div>

      {/* Column Information */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Detected Columns</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {file.columns.map((column) => (
            <div
              key={column}
              className="px-3 py-2 bg-gray-100 rounded-lg text-sm font-medium text-gray-700"
            >
              {column}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DataPreview;