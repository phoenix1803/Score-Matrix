import { NextResponse } from "next/server"
import { exec } from "child_process"
import fs from "fs"
import path from "path"
import { promisify } from "util"

const execPromise = promisify(exec)

export async function POST(request: Request) {
  try {
    const { filePath } = await request.json()
    if (!filePath) {
      return NextResponse.json(
        { error: "File path is required" },
        { status: 400 }
      )
    }

    const scriptPath = path.join(process.cwd(), "lib", "convert.py")
    const outputPath = path.join(process.cwd(), "public", "reports", "class_summary.xlsx")

    const outputDir = path.dirname(outputPath)
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true })
    }

    // Await the execution of the Python script
    try {
      await execPromise(`python ${scriptPath} ${filePath} ${outputPath}`)
    } catch (error) {
      console.error("Error converting JSON to Excel:", error)
      return NextResponse.json(
        { error: "Failed to convert JSON to Excel" },
        { status: 500 }
      )
    }

    const fileBuffer = fs.readFileSync(outputPath)

    return new NextResponse(fileBuffer, {
      headers: {
        "Content-Disposition": 'attachment; filename="report.xlsx"',
        "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      },
    })
  } catch (error) {
    console.error("Error in API route:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
