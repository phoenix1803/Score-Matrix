package com.example.scorematrix

import android.content.Context
import android.graphics.Bitmap
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.util.Log

import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.generationConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.google.ai.client.generativeai.type.Content
import com.google.ai.client.generativeai.type.content
import androidx.core.graphics.createBitmap


class GeminiEvaluator(private val context: Context) {


    private val generativeModel = GenerativeModel(
        modelName = "gemini-1.5-flash",
        apiKey = "xxxxxxxxx",
        generationConfig = generationConfig {
            temperature = 0.7f
            topK = 32
            topP = 1f
            maxOutputTokens = 4096
        }
    )
    suspend fun evaluatePdfs(
        questionPdfUri: Uri,
        answerPdfUri: Uri,
        parameters: Map<String, Float>
    ): String {
        return try {
            val questionBitmaps = renderPdfToBitmaps(questionPdfUri)
            val answerBitmaps = renderPdfToBitmaps(answerPdfUri)

            if (questionBitmaps.isEmpty() || answerBitmaps.isEmpty()) {
                return "Error: Could not extract images from PDFs"
            }

            val prompt = buildPrompt(questionBitmaps, answerBitmaps, parameters)
            val response = generativeModel.generateContent(prompt)

            response.text ?: "No evaluation result generated"
        } catch (e: Exception) {
            Log.e("GeminiEvaluator", "Evaluation error", e)
            "Evaluation failed: ${e.localizedMessage}"
        }
    }

    private fun buildPrompt(
        questionBitmaps: List<Bitmap>,
        answerBitmaps: List<Bitmap>,
        parameters: Map<String, Float>
    ): Content {
        return content {
            text(
                """
           You are an expert teacher evaluating a student's answer.


            Use the following evaluation parameters (scaled from 0–100):
            ${parameters.entries.joinToString("\n") { "${it.key}: ${it.value.toInt()}%" }}

            First, extract text via OCR from both the question paper and answer sheet. Then grade the answers according to these parameters. Be strict with low scores and generous with high ones. 
            
            Provide:
            - Total Marks
            - Marks Obtained
            - A clear explanation per answer
            - Suggestions for improvement in plain, unformatted text.
            Be to the point, do not give any clarification about the text being given by ocr or related to it and be confident, dont use workds like i assume, seems, etc..
            For the formatting:
            first give the total marks and marks obtained ( in bold and large text)
            then give question wise analysis of the answer written ( do not talk about the parameters like correctness, coherence, steps, etc. )
            then give a detailed summary of the overall answer sheet, what all student knows and where the student should work, what are his/her's weak points..
            """.trimIndent()
            )

            text("Here is the question paper:")
            questionBitmaps.forEach { image(it) }

            text("Here is the answer sheet:")
            answerBitmaps.forEach { image(it) }
        }
    }


    private fun renderPdfToBitmaps(pdfUri: Uri): List<Bitmap> {
        val bitmaps = mutableListOf<Bitmap>()
        val fileDescriptor = context.contentResolver.openFileDescriptor(pdfUri, "r") ?: return emptyList()
        val pdfRenderer = PdfRenderer(fileDescriptor)

        for (i in 0 until pdfRenderer.pageCount) {
            val page = pdfRenderer.openPage(i)
            val bitmap = createBitmap(page.width, page.height)
            page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
            bitmaps.add(bitmap)
            page.close()
        }

        pdfRenderer.close()
        fileDescriptor.close()
        return bitmaps
    }
}









