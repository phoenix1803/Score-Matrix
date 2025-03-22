import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';

export async function POST(request: Request): Promise<Response> {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const parameters = formData.get('parameters')?.toString() || "";

    if (!file) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 });
    }

    const uploadDir = path.join(process.cwd(), 'uploads');
    await fs.mkdir(uploadDir, { recursive: true });

    const ext = file.name ? path.extname(file.name) : '.txt'; // Ensure extension
    const filePath = path.join(uploadDir, `question-paper${ext}`);

    const buffer = Buffer.from(await file.arrayBuffer());
    await fs.writeFile(filePath, buffer);

    const referenceData = { filePath, parameters };
    await fs.writeFile(path.join(uploadDir, 'reference.json'), JSON.stringify(referenceData, null, 2));

    return NextResponse.json({ message: 'Reference file stored successfully', referenceData }, { status: 200 });
  } catch (error) {
    console.error("Error in file upload:", error);
    return NextResponse.json({ error: 'Internal server error', details: (error as Error).message }, { status: 500 });
  }
}
