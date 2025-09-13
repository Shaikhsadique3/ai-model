export type ProcessingState = 'idle' | 'uploaded' | 'processing' | 'completed' | 'error';

export interface UploadedFile {
  file_id: string;
  filename: string;
  preview_data: any[];
  columns: string[];
  total_rows: number;
}

export interface PredictionResults {
  total_customers: number;
  risk_distribution: {
    high_risk_count: number;
    medium_risk_count: number;
    low_risk_count: number;
    high_risk_percent: number;
    medium_risk_percent: number;
    low_risk_percent: number;
  };
  top_churn_reasons: Array<{
    reason: string;
    count: number;
    percentage: number;
  }>;
  average_churn_score: number;
  benchmark_comparison: {
    your_churn_rate: number;
    industry_average: number;
    performance: 'above' | 'below' | 'average';
  };
}

export interface FileUploadProps {
  onFileUploaded: (file: UploadedFile) => void;
}

export interface DataPreviewProps {
  file: UploadedFile;
  onGenerateReport: () => void;
}

export interface ProcessingStatusProps {
  state: ProcessingState;
  onRetry: () => void;
}

export interface ResultsSummaryProps {
  results: PredictionResults;
  reportUrl: string | null;
}