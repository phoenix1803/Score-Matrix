import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
  const outputDir = path.join(process.cwd(), 'outputs');

  // Ensure the directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const files = fs.readdirSync(outputDir);

  if (files.length === 0) {
    return NextResponse.json({ error: 'No files available for download' }, { status: 404 });
  }

  const filePath = path.join(outputDir, files[0]);
  const fileStream = fs.createReadStream(filePath);

  const readableStream = new ReadableStream({
    start(controller) {
      fileStream.on('data', chunk => {
        controller.enqueue(chunk);
      });

      fileStream.on('end', () => {
        controller.close();
      });

      fileStream.on('error', err => {
        controller.error(err);
      });
    },
  });

  return new NextResponse(readableStream, {
    headers: {
      'Content-Type': 'application/octet-stream',
      'Content-Disposition': `attachment; filename="${files[0]}"`,
    },
  });
}