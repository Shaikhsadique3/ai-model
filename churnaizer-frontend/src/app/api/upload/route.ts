import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return NextResponse.json({ error: 'No file uploaded.' }, { status: 400 });
    }

    // In a real application, you would process the file here.
    // For now, we'll just log its name and size.
    console.log(`Received file: ${file.name}, size: ${file.size} bytes`);

    // You would typically save the file or pass it to a backend service for processing.
    // For example, if you have a Python backend (like your churnaizer-api),
    // you would send this file to that backend.

    return NextResponse.json({ message: 'File uploaded successfully (placeholder).' });
  } catch (error) {
    console.error('Error uploading file:', error);
    return NextResponse.json({ error: 'Internal server error.' }, { status: 500 });
  }
}