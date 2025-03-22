import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import { spawn } from 'child_process';
import { existsSync } from 'fs';

export async function POST(request: Request): Promise<Response> {
  try {
    const formData = await request.formData();
    const files = formData.getAll('files') as File[];

    if (!files || files.length === 0) {
      return NextResponse.json({ error: 'No files uploaded' }, { status: 400 });
    }

    const uploadDir = path.join(process.cwd(), 'uploads');
    await fs.mkdir(uploadDir, { recursive: true });

    const filePaths: string[] = [];

    await Promise.all(
      files.map(async (file) => {
        const filePath = path.join(uploadDir, file.name);
        const buffer = Buffer.from(await file.arrayBuffer());
        await fs.writeFile(filePath, buffer);
        filePaths.push(filePath);
      })
    );

    return new Promise<Response>((resolve) => {  // Explicitly setting <Response>
      const venvPythonPath = "C:\\Users\\Admin\\anaconda3\\envs\\myenv\\python.exe";
      const pythonPath = existsSync(venvPythonPath) ? venvPythonPath : "python"; // Fallback

      const pythonProcess = spawn(pythonPath, [
        path.join(process.cwd(), 'lib', 'processFiles.py'),
        JSON.stringify(filePaths)
      ]);

      let output = '';
      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        console.error(`Python script error: ${data}`);
      });

      pythonProcess.on('close', (code) => {
        console.log(`Python script exited with code ${code}`);
        resolve(NextResponse.json({ message: 'Processing complete', output }, { status: 200 }));
      });
    });
  } catch (error) {
    return NextResponse.json({ error: 'Internal server error', details: (error as Error).message }, { status: 500 });
  }
}
