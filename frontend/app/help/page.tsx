"use client"

import React from 'react';
import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ChevronDown } from "lucide-react"

const faqs = [
  {
    question: "How do I upload files?",
    answer:
      "To upload files, go to the Upload page and follow the step-by-step instructions. You can upload a reference file and then multiple files for processing.",
  },
  {
    question: "What file formats are supported?",
    answer: "We support PDF, DOC, and DOCX file formats for processing.",
  },
  {
    question: "How long does processing take?",
    answer:
      "Processing time depends on the number and size of files. Typically, it takes a few minutes for small files and up to an hour for larger batches.",
  },
  {
    question: "Can I customize the processing parameters?",
    answer:
      "Yes, you can adjust the processing parameters such as Correctness, Grammar, and Steps on the Upload page before submitting your files.",
  },
  {
    question: "How do I access my processed files?",
    answer:
      "After processing is complete, you can download your files from the Thank You page or access them later from your account dashboard.",
  },
]

const FAQItem = ({ question, answer }: { question: string; answer: string }) => {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <motion.div
      initial={false}
      animate={{ backgroundColor: isOpen ? "rgba(229, 231, 235, 0.5)" : "transparent" }}
      className="border-b border-gray-200"
    >
      <button onClick={() => setIsOpen(!isOpen)} className="flex justify-between items-center w-full py-4 text-left">
        <span className="text-lg font-medium">{question}</span>
        <ChevronDown className={`w-5 h-5 transition-transform ${isOpen ? "transform rotate-180" : ""}`} />
      </button>
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            key="content"
            initial="collapsed"
            animate="open"
            exit="collapsed"
            variants={{
              open: { opacity: 1, height: "auto" },
              collapsed: { opacity: 0, height: 0 },
            }}
            transition={{ duration: 0.3, ease: [0.04, 0.62, 0.23, 0.98] }}
          >
            <div className="pb-4 text-gray-600">{answer}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

const HelpPage = () => {
  return (
    <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-white shadow rounded-lg p-8"
      >
        <h2 className="text-2xl font-bold mb-6">Frequently Asked Questions</h2>
        <div className="space-y-4">
          {faqs.map((faq, index) => (
            <FAQItem key={index} question={faq.question} answer={faq.answer} />
          ))}
        </div>
      </motion.div>
    </div>
  )
}

export default HelpPage

