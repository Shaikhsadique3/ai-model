'use client';

import { useState } from 'react';

export default function UploadForm() {
  const [email, setEmail] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage('');

    if (!email || !file) {
      setMessage('Please fill in all fields.');
      return;
    }

    if (!file.name.endsWith('.csv')) {
      setMessage('Please upload a CSV file.');
      return;
    }

    const formData = new FormData();
    formData.append('email', email);
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setMessage('Upload successful! ' + data.message);
      } else {
        setMessage('Upload failed: ' + data.detail || data.message);
      }
    } catch (error) {
      setMessage('An error occurred during upload.');
      console.error('Upload error:', error);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="email" className="block text-sm font-medium text-gray-700">
          Email
        </label>
        <input
          type="email"
          id="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
        />
      </div>
      <div>
        <label htmlFor="file" className="block text-sm font-medium text-gray-700">
          Upload CSV File
        </label>
        <input
          type="file"
          id="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files ? e.target.files[0] : null)}
          required
          className="mt-1 block w-full text-gray-700"
        />
      </div>
      <button
        type="submit"
        className="inline-flex justify-center rounded-md border border-transparent bg-indigo-600 py-2 px-4 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
      >
        Upload and Analyze
      </button>
      {message && <p className="mt-2 text-sm text-red-600">{message}</p>}
    </form>
  );
}