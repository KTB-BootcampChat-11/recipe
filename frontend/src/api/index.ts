import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
});

export interface AnalyzeResponse {
  job_id: string;
  message: string;
}

export interface JobStatus {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  video_id?: string;
}

export interface Ingredient {
  name: string;
  amount: string;
  unit: string;
  note?: string;
}

export interface Step {
  step_number: number;
  instruction: string;
  timestamp: number;
  duration?: string;
  tips?: string;
}

export interface Recipe {
  title: string;
  description?: string;
  servings?: string;
  total_time?: string;
  difficulty?: string;
  ingredients: Ingredient[];
  steps: Step[];
  tips?: string[];
}


export interface VideoInfo {
  video_id: string;
  title: string;
  duration: number;
  url: string;
}

export interface AnalysisResult {
  recipe: Recipe;
  video_info: VideoInfo;
  transcript: {
    full_text: string;
    segments: Array<{
      start: number;
      end: number;
      text: string;
    }>;
  };
}

export const analyzeVideo = async (url: string): Promise<AnalyzeResponse> => {
  const response = await api.post('/analyze', { url });
  return response.data;
};

export const getJobStatus = async (jobId: string): Promise<JobStatus> => {
  const response = await api.get(`/status/${jobId}`);
  return response.data;
};

export const getResult = async (jobId: string): Promise<AnalysisResult> => {
  const response = await api.get(`/result/${jobId}`);
  return response.data;
};

export default api;
