import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";

export async function POST(req: NextRequest): Promise<Response> {
  try {
    const { to, subject } = await req.json();

    if (!to || !subject) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 });
    }

    return new Promise((resolve) => {
      const pythonProcess = spawn("python", ["../frontend/lib/send_email.py", to, subject]);

      pythonProcess.stdout.on("data", (data) => {
        console.log(`Python Output: ${data}`);
      });

      pythonProcess.stderr.on("data", (data) => {
        console.error(`Python Error: ${data}`);
      });

      pythonProcess.on("close", (code) => {
        if (code === 0) {
          resolve(NextResponse.json({ message: "Email process triggered successfully" }, { status: 200 }));
        } else {
          resolve(NextResponse.json({ error: "Failed to send email" }, { status: 500 }));
        }
      });
    });
  } catch (error) {
    console.error("Server Error:", error);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
