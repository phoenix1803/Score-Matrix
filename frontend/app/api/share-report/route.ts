import { NextResponse } from "next/server"
import path from "path"

export async function POST(request: Request) {
  try {
    const { filePath } = await request.json()

    if (!filePath) {
      return NextResponse.json(
        { error: "File path is required" },
        { status: 400 }
      )
    }

    const shareableLink = `https://example.com/share/${path.basename(filePath)}`

    return NextResponse.json({ shareableLink }, { status: 200 })
  } catch (error) {
    console.error("Error in API route:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}