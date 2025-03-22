"use client"

import React from 'react';
import { useState, useEffect } from "react"
import { motion } from "framer-motion"

const ProfilePage = () => {
  const [user] = useState({
    name: "John Doe",
    email: "john@example.com",
    avatar: "/placeholder.svg",
  },)

  useEffect(() => {
    // Fetch user data from API or local storage
  }, [])

  return (
    <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-white shadow rounded-lg p-8"
      >
        <div className="flex items-center space-x-6 mb-6">
          <img src={user.avatar || "/placeholder.svg"} alt="User avatar" className="w-24 h-24 rounded-full" />
          <div>
            <h2 className="text-2xl font-bold">{user.name}</h2>
            <p className="text-gray-600">{user.email}</p>
          </div>
        </div>
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-medium">Account Information</h3>
            <p className="text-gray-600">Update your account information and email preferences.</p>
          </div>
          <div className="border-t border-gray-200 pt-6">
            <h3 className="text-lg font-medium">Security</h3>
            <p className="text-gray-600">Manage your password and account security preferences.</p>
          </div>
          <div className="border-t border-gray-200 pt-6">
            <h3 className="text-lg font-medium">Privacy</h3>
            <p className="text-gray-600">Control your privacy settings and data usage preferences.</p>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default ProfilePage

