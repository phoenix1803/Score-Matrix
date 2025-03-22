import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";
import path from "path";

export async function POST(req: NextRequest) {
    try {
        const { message, sessionId } = await req.json();
        const pythonScriptPath = path.join(process.cwd(), "../frontend/lib/chatbot.py");

        const pythonProcess = spawn("python", [pythonScriptPath, sessionId, message]);

        let botResponse = "";
        let errorOutput = "";

        pythonProcess.stdout.on("data", (data) => {
            botResponse += data.toString();
        });

        pythonProcess.stderr.on("data", (data) => {
            errorOutput += data.toString();
        });

        return new Promise((resolve) => {
            pythonProcess.on("close", (code) => {
                if (code === 0) {
                    resolve(NextResponse.json({ response: botResponse.trim() }));
                } else {
                    console.error("Python script error:", errorOutput);
                    resolve(NextResponse.json({ response: "Error in Python script execution." }, { status: 500 }));
                }
            });
        });
    } catch (error) {
        console.error("Error:", error);
        return NextResponse.json({ response: "Internal server error" }, { status: 500 });
    }
}
