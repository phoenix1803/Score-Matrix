"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { v4 as uuidv4 } from "uuid";
import styled, { keyframes } from 'styled-components';


const ChatbotModal = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [message, setMessage] = useState("");
    const [messages, setMessages] = useState<{ text: string; sender: string }[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState("");

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const messageInputRef = useRef<HTMLInputElement>(null);

    // Hardcoded system prompt
    const systemPrompt = "You are a helpful assistant specialized in answering questions about technology and programming. Keep your responses concise and professional.";

    // Initialize session ID on component mount
    useEffect(() => {
        setSessionId(uuidv4());
    }, []);

    // Scroll to bottom when messages change
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Focus input when chat opens
    useEffect(() => {
        if (isOpen) {
            messageInputRef.current?.focus();
        }
    }, [isOpen]);

    const handleSendMessage = async () => {
        if (!message.trim()) return;

        // Add user message to the chat
        setMessages((prev) => [...prev, { text: message, sender: "user" }]);
        setMessage("");
        setIsLoading(true);

        // Call the API to get the bot's response
        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    message,
                    sessionId,
                    systemPrompt, // Pass the hardcoded system prompt
                }),
            });

            const data = await response.json();
            setMessages((prev) => [...prev, { text: data.response, sender: "bot" }]);
        } catch (error) {
            console.error("Error fetching bot response:", error);
            setMessages((prev) => [...prev, { text: "Sorry, something went wrong.", sender: "bot" }]);
        } finally {
            setIsLoading(false);
        }
    };
    const loading = keyframes`
    0% { width: 0; }
    100% { width: 100%; }
  `;

    const LoadingBar = styled.div`
    animation: ${loading} 2s infinite;
  `;

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    return (
        <div className="fixed bottom-4 right-4 z-50">
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 20, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 20, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                        className="bg-white rounded-lg shadow-xl w-96 h-[400px] flex flex-col overflow-hidden"
                    >
                        <div className="p-4 bg-gradient-to-r from-blue-600 to-indigo-700 text-white flex justify-between items-center">
                            <h2 className="text-lg font-semibold">Chatbot</h2>
                            <button
                                onClick={() => setIsOpen(false)}
                                className="text-white hover:text-gray-200"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                                </svg>
                            </button>
                        </div>
                        <div className="flex-1 p-4 overflow-y-auto bg-gray-50">
                            {messages.length === 0 ? (
                                <div className="flex flex-col items-center justify-center h-full text-gray-500">
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-3 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                                    </svg>
                                    <p className="text-center text-sm">
                                        Start a conversation! Your chat history will be preserved until you reset or refresh the page.
                                    </p>
                                </div>
                            ) : (
                                <>
                                    {messages.map((msg, index) => (
                                        <div
                                            key={index}
                                            className={`mb-4 flex ${msg.sender === "user" ? "justify-end" : "justify-start"
                                                }`}
                                        >
                                            {msg.sender === "bot" && (
                                                <div className="h-8 w-8 rounded-full bg-indigo-600 text-white flex items-center justify-center mr-2 flex-shrink-0">
                                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                                    </svg>
                                                </div>
                                            )}
                                            <div
                                                className={`max-w-[75%] p-3 rounded-lg ${msg.sender === "user"
                                                    ? "bg-blue-600 text-white rounded-br-none"
                                                    : "bg-white text-gray-800 border border-gray-200 rounded-bl-none shadow-sm"
                                                    }`}
                                            >
                                                <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
                                            </div>
                                            {msg.sender === "user" && (
                                                <div className="h-8 w-8 rounded-full bg-blue-500 text-white flex items-center justify-center ml-2 flex-shrink-0">
                                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                                    </svg>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                    <div ref={messagesEndRef} />
                                </>
                            )}
                        </div>
                        <div className="p-3 border-t border-gray-200 bg-white">
                            {isLoading && (
                                <div className="absolute top-0 left-0 right-0 h-1">
                                    <div className="h-full bg-blue-600 rounded-r-full animate-loading"></div>
                                </div>
                            )}
                            <div className="flex items-center space-x-2">
                                <input
                                    ref={messageInputRef}
                                    type="text"
                                    value={message}
                                    onChange={(e) => setMessage(e.target.value)}
                                    onKeyPress={handleKeyPress}
                                    className="flex-1 p-2 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    placeholder="Type a message..."
                                    disabled={isLoading}
                                />
                                <button
                                    onClick={handleSendMessage}
                                    disabled={isLoading || !message.trim()}
                                    className={`p-2 rounded-full ${isLoading || !message.trim()
                                        ? "bg-gray-300 text-gray-500"
                                        : "bg-blue-600 text-white hover:bg-blue-700"
                                        } transition-colors`}
                                >
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-4 rounded-full shadow-lg hover:shadow-xl transition-all transform hover:scale-105 flex items-center justify-center"
            >
                {isOpen ? (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                    </svg>
                )}
            </button>
            <LoadingBar className="animate-loading" />
        </div>
    );
};

export default ChatbotModal;