"use client"

import React from 'react';
import { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { motion, useScroll, useSpring, useAnimation, } from "framer-motion";

interface SectionRefs {
  [key: string]: HTMLElement | null;
}

const HomePage = () => {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001,
  });

  const [scrollProgress, setScrollProgress] = useState(0);
  void scrollProgress;
  const [activeSection, setActiveSection] = useState("");
  const sectionRefs = useRef<SectionRefs>({});
  void sectionRefs;

  // Refs for each section
  const heroRef = useRef(null);
  const stepsRef = useRef(null);
  const videoRef = useRef(null);
  const pricingRef = useRef(null);
  const whyUsRef = useRef(null);
  const communityRef = useRef(null);

  // Animation controls
  const stepsControls = useAnimation();
  const videoControls = useAnimation();
  const pricingControls = useAnimation();
  const whyUsControls = useAnimation();
  const communityControls = useAnimation();

  // Animation for floating elements
  const floatingAnimation = {
    y: [0, -10, 0],
    transition: {
      duration: 3,
      repeat: Infinity,
      ease: "easeInOut"
    }
  };
  void floatingAnimation;
  useEffect(() => {
    const handleScroll = () => {
      const totalScroll = window.scrollY;
      const windowHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scroll = totalScroll / windowHeight;
      setScrollProgress(scroll);
    };

    window.addEventListener("scroll", handleScroll);
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    const observerOptions: IntersectionObserverInit = {
      root: null,
      rootMargin: "-10% 0px -10% 0px",
      threshold: 0.3,
    };

    const observerCallback = (entries: IntersectionObserverEntry[]) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          setActiveSection(entry.target.id);

          switch (entry.target.id) {
            case "steps":
              stepsControls.start("visible");
              break;
            case "video":
              videoControls.start("visible");
              break;
            case "pricing":
              pricingControls.start("visible");
              break;
            case "why-us":
              whyUsControls.start("visible");
              break;
            case "community":
              communityControls.start("visible");
              break;
          }
        }
      });
    };

    const observer = new IntersectionObserver(observerCallback, observerOptions);

    if (heroRef.current) observer.observe(heroRef.current);
    if (stepsRef.current) observer.observe(stepsRef.current);
    if (videoRef.current) observer.observe(videoRef.current);
    if (pricingRef.current) observer.observe(pricingRef.current);
    if (whyUsRef.current) observer.observe(whyUsRef.current);
    if (communityRef.current) observer.observe(communityRef.current);

    return () => observer.disconnect();
  }, [stepsControls, videoControls, pricingControls, whyUsControls, communityControls]);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  const sectionVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
        staggerChildren: 0.2,
        delayChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.6 }
    }
  };

  return (
    <div className="relative overflow-hidden">
      <motion.div
        className="fixed top-0 left-0 right-0 h-2 bg-gradient-to-r from-purple-600 via-pink-500 to-blue-500 z-50"
        style={{ scaleX }}
      />

      <div className="fixed right-6 top-1/2 transform -translate-y-1/2 z-40 hidden md:block">
        {["hero", "steps", "video", "pricing", "why-us", "community"].map((id) => (
          <motion.div
            key={id}
            className={`w-4 h-4 mb-4 rounded-full cursor-pointer border-2 border-primary ${activeSection === id ? "bg-primary" : "bg-transparent"
              }`}
            onClick={() => scrollToSection(id)}
            whileHover={{ scale: 1.2 }}
            whileTap={{ scale: 0.9 }}
          />
        ))}
      </div>

      <motion.section
        id="hero"
        ref={heroRef}
        className="h-screen flex items-center justify-center relative overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-800 via-indigo-900 to-purple-900" />

          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute rounded-full bg-white bg-opacity-10"
              style={{
                width: Math.random() * 100 + 50,
                height: Math.random() * 100 + 50,
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                x: [0, Math.random() * 50 - 25],
                y: [0, Math.random() * 50 - 25],
              }}
              transition={{
                duration: Math.random() * 100 + 100,
                repeat: Infinity,
                repeatType: "reverse",
                ease: "easeInOut",
              }}
            />
          ))}
        </div>

        <div className="text-center z-10 px-4 sm:px-6 lg:px-8 relative">
          <motion.h1
            className="text-4xl md:text-7xl font-bold mb-6 text-white"
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            Welcome to <span className="text-transparent bg-clip-text bg-gradient-to-r from-pink-500 to-yellow-400">ScoreMatrix</span>
          </motion.h1>

          <motion.p
            className="text-xl md:text-2xl mb-8 text-white text-opacity-90"
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            Streamline answer sheet evaluation with powerful AI precision
          </motion.p>

          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link
                href="/login"
                className="bg-gradient-to-r from-pink-500 to-purple-600 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:shadow-lg transition duration-300 inline-block"
              >
                Get Started
              </Link>
            </motion.div>

            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link
                href="#video"
                className="bg-gradient-to-r from-pink-700 to-purple-700 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:shadow-lg transition duration-300 inline-block"
                onClick={(e) => {
                  e.preventDefault();
                  scrollToSection("video");
                }}
              >
                Watch Demo
              </Link>
            </motion.div>
          </motion.div>


        </div>
      </motion.section>

      <motion.section
        id="steps"
        ref={stepsRef}
        className="py-24 bg-white relative overflow-hidden"
        variants={sectionVariants}
        initial="hidden"
        animate={stepsControls}
      >
        <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-pink-200 to-pink-100 rounded-full opacity-20 transform translate-x-1/2 -translate-y-1/2"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-gradient-to-tr from-blue-200 to-purple-100 rounded-full opacity-20 transform -translate-x-1/2 translate-y-1/2"></div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <motion.h2
            className="text-3xl md:text-5xl font-bold mb-4 text-center"
            variants={itemVariants}
          >
            How It <span className="text-primary">Works</span>
          </motion.h2>

          <motion.p
            className="text-lg text-gray-600 max-w-3xl mx-auto text-center mb-16"
            variants={itemVariants}
          >
            Three simple steps to transform your answer sheet evaluation process
          </motion.p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
            <div className="hidden md:block absolute top-1/3 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 transform -translate-y-1/2 z-0"></div>

            {[
              {
                title: "Upload Files",
                description: "Securely upload your answer sheets in bulk with our drag-and-drop interface",
                icon: (
                  <svg className="w-10 h-10 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                  </svg>
                )
              },
              {
                title: "AI Processing",
                description: "Our advanced algorithms analyze and grade each answer according to your criteria",
                icon: (
                  <svg className="w-10 h-10 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                  </svg>
                )
              },
              {
                title: "Get Results",
                description: "Download comprehensive reports and analytics in seconds, not days",
                icon: (
                  <svg className="w-10 h-10 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01"></path>
                  </svg>
                )
              },
            ].map((step, index) => (
              <motion.div
                key={step.title}
                className="bg-white p-8 rounded-2xl shadow-xl relative z-10"
                variants={itemVariants}
                whileHover={{
                  scale: 1.05,
                  boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
                  y: -10
                }}
              >
                <div className="flex items-center justify-center w-16 h-16 mb-6 rounded-full bg-primary bg-opacity-10 mx-auto">
                  {step.icon}
                </div>
                <h3 className="text-2xl font-semibold mb-4 text-center">{step.title}</h3>
                <p className="text-gray-600 text-center">{step.description}</p>
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center text-white font-bold text-lg z-20">
                  {index + 1}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.section>

      <motion.section
        id="video"
        ref={videoRef}
        className="py-24 bg-gradient-to-br from-indigo-900 to-purple-900 text-white relative overflow-hidden"
        variants={sectionVariants}
        initial="hidden"
        animate={videoControls}
      >
        {[...Array(15)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute rounded-full bg-white bg-opacity-5"
            style={{
              width: Math.random() * 150 + 50,
              height: Math.random() * 150 + 50,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.1, 0.2, 0.1],
            }}
            transition={{
              duration: Math.random() * 8 + 5,
              repeat: Infinity,
              repeatType: "reverse",
              ease: "easeInOut",
            }}
          />
        ))}

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <motion.h2
            className="text-3xl md:text-5xl font-bold mb-4 text-center"
            variants={itemVariants}
          >
            See <span className="text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-yellow-300">ScoreMatrix</span> in Action
          </motion.h2>

          <motion.p
            className="text-lg text-white text-opacity-80 max-w-3xl mx-auto text-center mb-12"
            variants={itemVariants}
          >
            Watch how our platform revolutionizes the answer sheet evaluation process
          </motion.p>

          <motion.div
            className="relative mx-auto rounded-xl overflow-hidden shadow-2xl max-w-4xl"
            variants={itemVariants}
            whileHover={{ scale: 1.02 }}
          >
            <div className="bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500 p-1 rounded-xl">
              <iframe
                src="https://www.youtube.com/embed/oQf9VVYP4KE"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                className="w-full aspect-video rounded-lg"
              ></iframe>
            </div>
          </motion.div>
        </div>
      </motion.section>

      <motion.section
        id="pricing"
        ref={pricingRef}
        className="py-24 bg-white relative overflow-hidden"
        variants={sectionVariants}
        initial="hidden"
        animate={pricingControls}
      >
        <div className="absolute top-0 left-1/4 w-64 h-64 bg-gradient-to-br from-purple-200 to-indigo-100 rounded-full opacity-20"></div>
        <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-gradient-to-tr from-pink-200 to-red-100 rounded-full opacity-20"></div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <motion.h2
            className="text-3xl md:text-5xl font-bold mb-4 text-center"
            variants={itemVariants}
          >
            Simple <span className="text-primary">Pricing</span>
          </motion.h2>

          <motion.p
            className="text-lg text-gray-600 max-w-3xl mx-auto text-center mb-16"
            variants={itemVariants}
          >
            Choose the plan that works best for you!
          </motion.p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                title: "Small-Scale Institution",
                price: "₹7.50",
                unit: "per booklet",
                subtitle: "(20 pages)",
                description: "Perfect for schools and small colleges",
                features: [
                  "Process up to 50 PDFs",
                  "Fast Processing",
                  "Summary insights & performance trends",
                  "All answers as consolidated PDF",
                  "Email support"
                ],
                isPopular: false,
                cta: "Get Started"
              },
              {
                title: "Growing Institution",
                price: "₹12",
                unit: "per booklet",
                subtitle: "(20 pages)",
                description: "Ideal for medium-sized institutions",
                features: [
                  "Process up to 250 PDFs",
                  "Priority Processing",
                  "Detailed answer breakdown with AI feedback",
                  "Automated plagiarism check",
                  "Phone & email support",
                  "Multi-user collaborations"
                ],
                isPopular: true,
                cta: "Get Started"
              },
              {
                title: "Enterprise",
                price: "₹18",
                unit: "per booklet",
                subtitle: "(20 pages)",
                description: "For large educational institutions",
                features: [
                  "Unlimited PDF processing",
                  "Instant evaluation",
                  "AI-powered deep insights & predictive analytics",
                  "Multi-institution access & centralized data management",
                  "Dedicated account manager",
                  "Seamless API integration with third-party tools and websites",
                  "Pay as you go flexibility"
                ],
                isPopular: false,
                cta: "Get Started"
              },
            ].map((plan) => (
              <motion.div
                key={plan.title}
                className={`relative bg-white p-8 rounded-2xl border ${plan.isPopular
                  ? "border-purple-400 shadow-xl"
                  : "border-gray-200 shadow-lg"
                  }`}
                variants={itemVariants}
                whileHover={{
                  scale: 1.03,
                  boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
                  y: -10
                }}
              >
                {plan.isPopular && (
                  <div className="absolute top-0 right-0 bg-gradient-to-r from-purple-600 to-indigo-600 text-white text-sm font-bold px-4 py-1 rounded-bl-lg rounded-tr-xl transform translate-y-0 shadow-md">
                    Most Popular
                  </div>
                )}

                <h3 className="text-xl font-bold mb-2">{plan.title}</h3>
                <p className="text-gray-600 text-sm mb-6">{plan.description}</p>

                <div className="flex items-end mb-6">
                  <div className="text-4xl font-bold">{plan.price}</div>
                  <div className="ml-2 text-gray-600 pb-1">
                    <div>{plan.unit}</div>
                    <div className="text-sm">{plan.subtitle}</div>
                  </div>
                </div>

                <ul className="space-y-3 mb-8">
                  {plan.features.map((feature) => (
                    <li key={feature} className="flex items-start">
                      <svg
                        className="w-5 h-5 mr-2 text-green-500 mt-1 flex-shrink-0"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                      <span className="text-gray-700">{feature}</span>
                    </li>
                  ))}
                </ul>

                <Link
                  href="/signup"
                  className={`w-full block text-center py-3 rounded-lg font-medium transition duration-300 mt-auto ${plan.isPopular
                    ? "bg-gradient-to-r from-purple-600 to-indigo-600 text-white hover:from-purple-700 hover:to-indigo-700"
                    : "bg-gray-100 text-gray-800 hover:bg-gray-200"
                    }`}
                >
                  {plan.cta}
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.section>

      <motion.section
        id="why-us"
        ref={whyUsRef}
        className="py-24 bg-gradient-to-br from-blue-900 to-indigo-900 text-white relative overflow-hidden"
        variants={sectionVariants}
        initial="hidden"
        animate={whyUsControls}
      >
        <div className="absolute inset-0 overflow-hidden">
          {[...Array(5)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute bg-white bg-opacity-5 rounded-full"
              style={{
                width: Math.random() * 300 + 100,
                height: Math.random() * 300 + 100,
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                x: [0, Math.random() * 50 - 25],
                y: [0, Math.random() * 50 - 25],
                rotate: [0, 360],
              }}
              transition={{
                duration: Math.random() * 20 + 20,
                repeat: Infinity,
                repeatType: "reverse",
                ease: "easeInOut",
              }}
            />
          ))}
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <motion.h2
            className="text-3xl md:text-5xl font-bold mb-4 text-center"
            variants={itemVariants}
          >
            Why Choose <span className="text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-yellow-300">ScoreMatrix</span>?
          </motion.h2>

          <motion.p
            className="text-lg text-white text-opacity-80 max-w-3xl mx-auto text-center mb-16"
            variants={itemVariants}
          >
            Join thousands of educational institutions that have transformed their evaluation process
          </motion.p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {[
              {
                title: "Advanced AI Technology",
                description: "Our proprietary algorithms ensure accurate and consistent grading across all subjects and question types.",
                icon: (
                  <svg className="w-10 h-10 text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                  </svg>
                )
              },
              {
                title: "Secure & Confidential",
                description: "Bank-level encryption and strict data protocols ensure your students' information remains protected.",
                icon: (
                  <svg className="w-10 h-10 text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path>
                  </svg>
                )
              },
              {
                title: "Time & Cost Efficient",
                description: "Reduce evaluation time by up to 70% and save on administrative costs with our automated solution.",
                icon: (
                  <svg className="w-10 h-10 text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                )
              },
              {
                title: "24/7 Expert Support",
                description: "Our organization is available around the clock to assist with any questions.",
                icon: (
                  <svg className="w-10 h-10 text-blue-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M18.364 5.636l-3.536 3.536m0 5.656l3.536 3.536M5.636 5.636l3.536 3.536m0 5.656l-3.536 3.536M21 12h-2m-4 0h-2m4-6h2m-8 0H9m8 12h2m-8 0H9"></path>
                  </svg>
                )
              },
            ].map((feature) => (
              <motion.div
                key={feature.title}
                className="bg-white bg-opacity-10 p-8 rounded-2xl backdrop-blur-sm border border-white border-opacity-10"
                variants={itemVariants}
                whileHover={{
                  scale: 1.05,
                  backgroundColor: "rgba(255, 255, 255, 0.15)",
                  boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)",
                }}
              >
                <div className="flex items-center justify-center w-16 h-16 mb-6 rounded-full bg-white bg-opacity-10 mx-auto">
                  {feature.icon}
                </div>
                <h3 className="text-2xl font-semibold mb-4 text-center">{feature.title}</h3>
                <p className="text-white text-opacity-80 text-center">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.section>

      <motion.section
        id="community"
        ref={communityRef}
        className="py-24 bg-white relative overflow-hidden"
        variants={sectionVariants}
        initial="hidden"
        animate={communityControls}
      >
        <div className="absolute top-0 left-1/4 w-64 h-64 bg-gradient-to-br from-purple-200 to-indigo-100 rounded-full opacity-20"></div>
        <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-gradient-to-tr from-pink-200 to-red-100 rounded-full opacity-20"></div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <motion.h2
            className="text-3xl md:text-5xl font-bold mb-4 text-center"
            variants={itemVariants}
          >
            Join Our <span className="text-primary">Community</span>
          </motion.h2>

          <motion.p
            className="text-lg text-gray-600 max-w-3xl mx-auto text-center mb-16"
            variants={itemVariants}
          >
            Join Us in Revolutionizing Evaluation Processes!
          </motion.p>

          <div className="flex flex-col items-center">
            <motion.div
              className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:shadow-lg transition duration-300 inline-block"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Link href="/community">
                Join Now
              </Link>
            </motion.div>
          </div>
        </div>
      </motion.section>
    </div>
  );
};

export default HomePage;