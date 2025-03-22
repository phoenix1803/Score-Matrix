import Link from "next/link"
import React from 'react';
const Footer = () => {
  return (
    <footer className="bg-gray-100 border-t border-gray-200">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="text-gray-500 text-sm">© 2025 Hyper Grey. All rights reserved.</div>
          <div className="flex space-x-6 mt-4 md:mt-0">
            <Link href="/privacy" className="text-gray-500 hover:text-gray-700 text-sm">
              Privacy Policy
            </Link>
            <Link href="/terms" className="text-gray-500 hover:text-gray-700 text-sm">
              Terms of Service
            </Link>
            <Link href="/contact" className="text-gray-500 hover:text-gray-700 text-sm">
              Contact Us
            </Link>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer

