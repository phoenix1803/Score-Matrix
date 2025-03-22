"use client"

import React from 'react';
import { useEffect, useState } from "react";
import { motion, useAnimation } from "framer-motion";
import Link from "next/link";

const SettingsPage = () => {
  const [settings, setSettings] = useState({
    notifications: true,
    darkMode: false,
    language: "en",
    autoSave: true,
    fontSize: "medium",
    emailFrequency: "daily",
    compactView: false,
    highContrast: false
  });

  const [resetConfirm, setResetConfirm] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const formControls = useAnimation();
  const buttonControls = useAnimation();

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5 }
    }
  };

  const buttonVariants = {
    hover: {
      scale: 1.05,
      boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
      transition: { duration: 0.3 }
    },
    tap: { scale: 0.95 }
  };

  useEffect(() => {
    formControls.start("visible");
    buttonControls.start("visible");
  }, [formControls, buttonControls]);

  const handleSettingChange = (setting: string, value: string | number | boolean) => {
    setSettings((prevSettings) => ({
      ...prevSettings,
      [setting]: value,
    }));

    // Show save success indicator temporarily
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 2000);
  };

  const handleResetToDefaults = () => {
    // Define default settings
    const defaultSettings = {
      notifications: true,
      darkMode: false,
      language: "en",
      autoSave: true,
      fontSize: "medium",
      emailFrequency: "daily",
      compactView: false,
      highContrast: false
    };

    // Animation sequence for resetting
    formControls.start({
      opacity: 0,
      y: 20,
      transition: { duration: 0.3 }
    }).then(() => {
      setSettings(defaultSettings);
      setResetConfirm(false);
      formControls.start({
        opacity: 1,
        y: 0,
        transition: { duration: 0.5 }
      });
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 relative overflow-hidden py-12 px-4 sm:px-6 lg:px-8">
      {/* Background elements */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-pink-200 to-pink-100 rounded-full opacity-20 transform translate-x-1/2 -translate-y-1/2"></div>
      <div className="absolute bottom-0 left-0 w-96 h-96 bg-gradient-to-tr from-blue-200 to-purple-100 rounded-full opacity-20 transform -translate-x-1/2 translate-y-1/2"></div>

      {/* Floating bubbles animation */}
      {[...Array(10)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute rounded-full bg-white bg-opacity-10"
          style={{
            width: Math.random() * 100 + 30,
            height: Math.random() * 100 + 30,
            border: "1px solid rgba(255, 255, 255, 0.2)",
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
          }}
          animate={{
            x: [0, Math.random() * 50 - 25],
            y: [0, Math.random() * 50 - 25],
          }}
          transition={{
            duration: Math.random() * 10 + 10,
            repeat: Infinity,
            repeatType: "reverse",
            ease: "easeInOut",
          }}
        />
      ))}

      {/* Progress indicator */}
      {saveSuccess && (
        <motion.div
          className="fixed top-4 right-4 bg-gradient-to-r from-green-500 to-green-600 text-white py-2 px-4 rounded-lg shadow-lg z-50"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
        >
          Settings saved successfully!
        </motion.div>
      )}

      <div className="max-w-4xl mx-auto relative z-10">
        <motion.div
          className="flex items-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Link
            href="/"
            className="flex items-center text-gray-600 hover:text-indigo-600 transition duration-300"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
            </svg>
            Back to Home
          </Link>
        </motion.div>

        <motion.div
          className="bg-white shadow-xl rounded-2xl p-8 border border-gray-100"
          variants={containerVariants}
          initial="hidden"
          animate={formControls}
        >
          <motion.div
            className="flex justify-between items-center mb-8 pb-4 border-b border-gray-100"
            variants={itemVariants}
          >
            <h2 className="text-3xl font-bold text-gray-800">
              Your <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-indigo-600">Settings</span>
            </h2>
            <div className="text-sm text-gray-500">Last updated: Today</div>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <motion.div
              className="space-y-8"
              variants={itemVariants}
            >
              <div className="bg-gray-50 p-6 rounded-xl shadow-sm">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">Notifications</h3>
                <div className="space-y-4">
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.notifications}
                      onChange={(e) => handleSettingChange("notifications", e.target.checked)}
                      className="form-checkbox h-5 w-5 text-indigo-600 rounded"
                    />
                    <span className="ml-3 text-gray-700">Enable notifications</span>
                  </label>

                  <div>
                    <p className="text-sm text-gray-600 mb-2">Email frequency</p>
                    <select
                      value={settings.emailFrequency}
                      onChange={(e) => handleSettingChange("emailFrequency", e.target.value)}
                      className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                      disabled={!settings.notifications}
                    >
                      <option value="daily">Daily</option>
                      <option value="weekly">Weekly</option>
                      <option value="monthly">Monthly</option>
                      <option value="never">Never</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 p-6 rounded-xl shadow-sm">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">Appearance</h3>
                <div className="space-y-4">
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.darkMode}
                      onChange={(e) => handleSettingChange("darkMode", e.target.checked)}
                      className="form-checkbox h-5 w-5 text-indigo-600 rounded"
                    />
                    <span className="ml-3 text-gray-700">Dark mode</span>
                  </label>

                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.compactView}
                      onChange={(e) => handleSettingChange("compactView", e.target.checked)}
                      className="form-checkbox h-5 w-5 text-indigo-600 rounded"
                    />
                    <span className="ml-3 text-gray-700">Compact view</span>
                  </label>

                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.highContrast}
                      onChange={(e) => handleSettingChange("highContrast", e.target.checked)}
                      className="form-checkbox h-5 w-5 text-indigo-600 rounded"
                    />
                    <span className="ml-3 text-gray-700">High contrast</span>
                  </label>
                </div>
              </div>
            </motion.div>

            <motion.div
              className="space-y-8"
              variants={itemVariants}
            >
              <div className="bg-gray-50 p-6 rounded-xl shadow-sm">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">Preferences</h3>
                <div className="space-y-4">
                  <div>
                    <p className="text-sm text-gray-600 mb-2">Language</p>
                    <select
                      value={settings.language}
                      onChange={(e) => handleSettingChange("language", e.target.value)}
                      className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                    >
                      <option value="en">English</option>
                      <option value="es">Español</option>
                      <option value="fr">Français</option>
                      <option value="de">Deutsch</option>
                      <option value="hi">हिन्दी</option>
                    </select>
                  </div>

                  <div>
                    <p className="text-sm text-gray-600 mb-2">Font Size</p>
                    <select
                      value={settings.fontSize}
                      onChange={(e) => handleSettingChange("fontSize", e.target.value)}
                      className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                    >
                      <option value="small">Small</option>
                      <option value="medium">Medium</option>
                      <option value="large">Large</option>
                      <option value="x-large">Extra Large</option>
                    </select>
                  </div>

                  <label className="flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.autoSave}
                      onChange={(e) => handleSettingChange("autoSave", e.target.checked)}
                      className="form-checkbox h-5 w-5 text-indigo-600 rounded"
                    />
                    <span className="ml-3 text-gray-700">Enable auto-save</span>
                  </label>
                </div>
              </div>

              <div className="bg-gray-50 p-6 rounded-xl shadow-sm">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">Reset Options</h3>

                {resetConfirm ? (
                  <div className="space-y-4">
                    <p className="text-sm text-red-600">Are you sure? This cannot be undone.</p>
                    <div className="flex space-x-4">
                      <motion.button
                        className="px-4 py-2 bg-red-500 text-white rounded-lg shadow"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={handleResetToDefaults}
                      >
                        Yes, Reset
                      </motion.button>
                      <motion.button
                        className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg shadow"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setResetConfirm(false)}
                      >
                        Cancel
                      </motion.button>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col sm:flex-row space-y-4 sm:space-y-0 sm:space-x-4">
                    <motion.button
                      className="px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-medium rounded-lg shadow hover:shadow-lg transition duration-300"
                      variants={buttonVariants}
                      whileHover="hover"
                      whileTap="tap"
                      onClick={() => setResetConfirm(true)}
                    >
                      Reset to Defaults
                    </motion.button>

                    <motion.button
                      className="px-6 py-3 bg-gradient-to-r from-gray-200 to-gray-300 text-gray-700 font-medium rounded-lg shadow hover:shadow-lg transition duration-300"
                      variants={buttonVariants}
                      whileHover="hover"
                      whileTap="tap"

                      onClick={() => {
                        // Mock export functionality
                        alert("Settings exported successfully!");
                      }}
                    >
                      Export Settings
                    </motion.button>
                  </div>
                )}
              </div>
            </motion.div>
          </div>

          {/* Profile section */}
          <motion.div
            className="mt-8 pt-8 border-t border-gray-100"
            variants={itemVariants}
          >
            <h3 className="text-lg font-semibold mb-4 text-gray-800">Profile Settings</h3>
            <div className="flex flex-col md:flex-row gap-8">
              <div className="flex-shrink-0">
                <div className="relative w-24 h-24 rounded-full overflow-hidden bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 p-1">
                  <div className="w-full h-full rounded-full overflow-hidden bg-white">
                    <img
                      src="/api/placeholder/100/100"
                      alt="User avatar"
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <button className="absolute bottom-0 right-0 bg-indigo-600 text-white rounded-full p-1 shadow-md hover:bg-indigo-700 transition duration-300">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"></path>
                    </svg>
                  </button>
                </div>
              </div>
              <div className="flex-grow space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                      Name
                    </label>
                    <input
                      type="text"
                      id="name"
                      className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                      placeholder="Your name"
                    />
                  </div>
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                      Email
                    </label>
                    <input
                      type="email"
                      id="email"
                      className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                      placeholder="your.email@example.com"
                    />
                  </div>
                </div>
                <div>
                  <label htmlFor="bio" className="block text-sm font-medium text-gray-700 mb-1">
                    Bio
                  </label>
                  <textarea
                    id="bio"
                    rows={3}
                    className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
                    placeholder="A short bio about yourself"
                  ></textarea>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Save button section */}
          <motion.div
            className="mt-8 pt-8 border-t border-gray-100 flex justify-end"
            variants={itemVariants}
          >
            <motion.button
              className="px-8 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white font-medium rounded-lg shadow-md hover:shadow-lg transition duration-300"
              variants={buttonVariants}
              whileHover="hover"
              whileTap="tap"
              onClick={() => {
                // Mock save functionality
                setSaveSuccess(true);
                setTimeout(() => setSaveSuccess(false), 2000);
              }}
            >
              Save Changes
            </motion.button>
          </motion.div>
        </motion.div>

        {/* Quick help section */}
        <motion.div
          className="mt-8 bg-white shadow-md rounded-xl p-6 border border-gray-100"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.8 }}
        >
          <div className="flex items-center">
            <svg className="w-5 h-5 text-indigo-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <h3 className="text-sm font-medium text-gray-700">Need help? Check out our <Link href="/help" className="text-indigo-600 hover:text-indigo-800 transition duration-300">help center</Link> or <Link href="/contact" className="text-indigo-600 hover:text-indigo-800 transition duration-300">contact support</Link>.</h3>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default SettingsPage;