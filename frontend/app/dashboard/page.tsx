"use client";

import React from 'react';
import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import ProtectedRoute from "../../components/ProtectedRoute";

interface Student {
  rollNumber: string;
  name: string;
  email: string;
  weakTopics: string;
}

interface Test {
  id: string;
  name: string;
  date: string;
  status: string;
  subject: string;
  totalMarks: number;
  uploadDate: string;
}

interface Book {
  id: string;
  name: string;
  path: string;
  uploadedAt: string;
}

interface LessonPlan {
  id: string;
  name: string;
  subject: string;
  topics: string[];
  duration: number;
  createdAt: string;
}

interface AssignmentOptions {
  topics: string;
  questionCount: number;
  difficulty: number;
  type: 'HW' | 'Test' | 'Assignment' | 'Custom';
  customType?: string;
  questionTypes: string[];
  totalMarks: number;
  marksDistribution: 'equal' | 'smart';
  duration?: number;
  bloomLevel?: string;
  includeAnswerKey: boolean;
  instructions?: string;
}

const Dashboard = ({ username }: { username: string }) => {
  const router = useRouter();
  const [students, setStudents] = useState<Student[]>([]);
  const [tests, setTests] = useState<Test[]>([
    {
      id: "1",
      name: "Test",
      date: "2025-01-03",
      status: "Evaluated",
      subject: "Applied Chemistry",
      totalMarks: 10,
      uploadDate: "2025-02-27"
    },
  ]);
  const [showAddStudentForm, setShowAddStudentForm] = useState(false);
  const [newStudent, setNewStudent] = useState<Student>({
    rollNumber: "",
    name: "",
    email: "",
    weakTopics: "-",
  });
  const [emailContent, setEmailContent] = useState("");
  const [selectedEmailStudent, setSelectedEmailStudent] = useState<Student | null>(null);
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [activeSection, setActiveSection] = useState<"students" | "tests" | "materials" | "lesson-plans" | "assignment-generator">("students");
  const [isLoading, setIsLoading] = useState(false);
  const [books, setBooks] = useState<Book[]>([]);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [showAddTestForm, setShowAddTestForm] = useState(false);
  const [newTest, setNewTest] = useState<Omit<Test, "id">>({
    name: "",
    date: "",
    status: "Pending",
    subject: "",
    totalMarks: 100,
    uploadDate: new Date().toISOString().split('T')[0]
  });
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState<{
    field: string;
    direction: "asc" | "desc";
  }>({ field: "", direction: "asc" });
  const [showNotification, setShowNotification] = useState(false);
  const [notificationMessage, setNotificationMessage] = useState("");
  const [lessonPlans, setLessonPlans] = useState<LessonPlan[]>([]);
  const [showLessonPlanForm, setShowLessonPlanForm] = useState(false);
  const [newLessonPlan, setNewLessonPlan] = useState({
    name: "",
    subject: "",
    topics: "",
    duration: 1
  });
  const [assignmentOptions, setAssignmentOptions] = useState<AssignmentOptions>({
    topics: "",
    questionCount: 10,
    difficulty: 3,
    type: 'HW',
    customType: '',
    questionTypes: ['MCQ'],
    totalMarks: 100,
    marksDistribution: 'equal',
    duration: 60,
    bloomLevel: 'Understanding',
    includeAnswerKey: true,
    instructions: ''
  });
  const [showAssignmentGenerator, setShowAssignmentGenerator] = useState(false);
  const [generatedContent, setGeneratedContent] = useState("");
  const [showContentModal, setShowContentModal] = useState(false);

  const showSuccessNotification = (message: string) => {
    setNotificationMessage(message);
    setShowNotification(true);
    setTimeout(() => {
      setShowNotification(false);
    }, 3000);
  };

  const handleCheckAnswerSheets = () => {
    router.push("/upload");
  };

  const handleAddStudent = () => {
    setShowAddStudentForm(true);
  };

  useEffect(() => {
    fetch("/api/students")
      .then(response => response.json())
      .then(data => {
        setStudents(data);
      })
      .catch(error => console.error("Error fetching students:", error));
  }, []);

  const handleSaveStudent = () => {
    if (!newStudent.rollNumber || !newStudent.name || !newStudent.email) {
      showSuccessNotification("Please fill all required fields");
      return;
    }

    setIsLoading(true);

    fetch("/api/students", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(newStudent),
    })
      .then(response => response.json())
      .then(data => {
        if (data.message === "Student added successfully!") {
          const updatedStudents = [...students, data.student];
          setStudents(updatedStudents);
          showSuccessNotification("Student added successfully!");
        } else {
          showSuccessNotification("Failed to add student");
        }
      })
      .catch(error => {
        console.error("Error:", error);
        showSuccessNotification("Failed to add student");
      })
      .finally(() => {
        setIsLoading(false);
        setNewStudent({ rollNumber: "", name: "", email: "", weakTopics: "-" });
        setShowAddStudentForm(false);
      });
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setNewStudent((prev) => ({ ...prev, [name]: value }));
  };

  const handleTestInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setNewTest((prev) => ({ ...prev, [name]: value }));
  };

  const handleLessonPlanInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setNewLessonPlan(prev => ({ ...prev, [name]: value }));
  };

  const handleAssignmentOptionChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target;

    if (type === 'checkbox') {
      const target = e.target as HTMLInputElement;
      if (name === 'questionTypes') {
        setAssignmentOptions(prev => {
          const newTypes = target.checked
            ? [...prev.questionTypes, value]
            : prev.questionTypes.filter(t => t !== value);
          return { ...prev, questionTypes: newTypes };
        });
      } else {
        setAssignmentOptions(prev => ({ ...prev, [name]: target.checked }));
      }
    } else {
      setAssignmentOptions(prev => ({ ...prev, [name]: value }));
    }
  };

  const handleGenerateLessonPlan = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/lesson-plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...newLessonPlan,
          topics: newLessonPlan.topics.split(',').map(topic => topic.trim())
        })
      });
      const data = await response.json();
      setGeneratedContent(data.content);
      setShowContentModal(true);
      showSuccessNotification("Lesson plan generated successfully!");
    } catch (error) {
      showSuccessNotification("Failed to generate lesson plan");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateAssignment = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/assignment-generator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(assignmentOptions)
      });
      const data = await response.json();
      setGeneratedContent(data.content);
      setShowContentModal(true);
      showSuccessNotification("Assignment generated successfully!");
    } catch (error) {
      showSuccessNotification("Failed to generate assignment");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendEmail = () => {
    if (!selectedEmailStudent || !emailContent) {
      showSuccessNotification("Please fill email Subject");
      return;
    }

    setIsLoading(true);
    fetch("/api/sendEmail", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        to: selectedEmailStudent.email,
        subject: emailContent,
      }),
    })
      .then(response => {
        if (response.ok) {
          showSuccessNotification(`Email sent to ${selectedEmailStudent.name} successfully!`);
        } else {
          showSuccessNotification("Failed to send email");
        }
      })
      .catch(error => {
        console.error("Error:", error);
        showSuccessNotification("Failed to send email");
      })
      .finally(() => {
        setIsLoading(false);
        setSelectedEmailStudent(null);
        setEmailContent("");
        setShowEmailModal(false);
      });
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      showSuccessNotification("Please select a file to upload");
      return;
    }

    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/api/uploadMaterial", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        const newBook = {
          id: Date.now().toString(),
          name: selectedFile.name,
          path: data.path,
          uploadedAt: new Date().toISOString(),
        };
        setBooks((prevBooks) => [...prevBooks, newBook]);
        showSuccessNotification("File uploaded successfully!");
      } else {
        showSuccessNotification("Failed to upload file");
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      showSuccessNotification("Failed to upload file");
    } finally {
      setIsLoading(false);
      setSelectedFile(null);
      setShowUploadModal(false);
    }
  };

  useEffect(() => {
    const fetchMaterials = async () => {
      const response = await fetch("/api/getMaterials");
      if (response.ok) {
        const data = await response.json();
        setBooks(data);
      }
    };

    if (activeSection === "materials") {
      fetchMaterials();
    }
  }, [activeSection]);

  const handleAddTest = () => {
    if (!newTest.name || !newTest.date || !newTest.subject) {
      showSuccessNotification("Please fill all required fields");
      return;
    }

    setIsLoading(true);
    setTimeout(() => {
      const newTestWithId = {
        ...newTest,
        id: (tests.length + 1).toString(),
      };

      setTests([...tests, newTestWithId]);

      fetch("/api/tests", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newTestWithId),
      })
        .then(response => {
          if (response.ok) {
            showSuccessNotification("Test added successfully!");
          } else {
            showSuccessNotification("Failed to add test");
          }
        })
        .catch(error => {
          console.error("Error:", error);
          showSuccessNotification("Failed to add test");
        })
        .finally(() => {
          setIsLoading(false);
          setNewTest({
            name: "",
            date: "",
            status: "Pending",
            subject: "",
            totalMarks: 100,
            uploadDate: new Date().toISOString().split('T')[0]
          });
          setShowAddTestForm(false);
        });
    }, 500);
  };

  const openEmailModal = (student: Student) => {
    setSelectedEmailStudent(student);
    setShowEmailModal(true);
  };

  const handleSort = (field: string) => {
    if (activeSection === "students" && (field === "rollNumber" || field === "name" || field === "email")) {
      setSortBy({
        field,
        direction: sortBy.field === field && sortBy.direction === "asc" ? "desc" : "asc",
      });
    } else if (activeSection === "tests" && (field === "name" || field === "date" || field === "subject" || field === "status" || field === "totalMarks")) {
      setSortBy({
        field,
        direction: sortBy.field === field && sortBy.direction === "asc" ? "desc" : "asc",
      });
    }
  };

  const getSortedData = () => {
    if (activeSection === "students" && sortBy.field) {
      return [...students].sort((a, b) => {
        const aValue = a[sortBy.field as keyof Student];
        const bValue = b[sortBy.field as keyof Student];

        if (sortBy.direction === "asc") {
          return aValue > bValue ? 1 : -1;
        } else {
          return aValue < bValue ? 1 : -1;
        }
      });
    }

    if (activeSection === "tests" && sortBy.field) {
      return [...tests].sort((a, b) => {
        const aValue = a[sortBy.field as keyof Test];
        const bValue = b[sortBy.field as keyof Test];

        if (sortBy.direction === "asc") {
          return aValue > bValue ? 1 : -1;
        } else {
          return aValue < bValue ? 1 : -1;
        }
      });
    }

    return activeSection === "students" ? students : tests;
  };

  const getFilteredData = (): Student[] | Test[] => {
    const sorted = getSortedData();

    if (!searchTerm) return sorted;

    if (activeSection === "students") {
      return (sorted as Student[]).filter(student =>
        student.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        student.rollNumber.toLowerCase().includes(searchTerm.toLowerCase()) ||
        student.email.toLowerCase().includes(searchTerm.toLowerCase())
      );
    } else {
      return (sorted as Test[]).filter(test =>
        test.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        test.subject.toLowerCase().includes(searchTerm.toLowerCase()) ||
        test.status.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
  };

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-500 p-4 md:p-8">
        <AnimatePresence>
          {showNotification && (
            <motion.div
              initial={{ opacity: 0, y: -50, x: "-50%" }}
              animate={{ opacity: 1, y: 20, x: "-50%" }}
              exit={{ opacity: 0, y: -50 }}
              className="fixed top-0 left-1/2 transform z-50 bg-white text-gray-800 py-2 px-4 rounded-lg shadow-lg"
            >
              {notificationMessage}
            </motion.div>
          )}
        </AnimatePresence>

        <motion.div
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="max-w-6xl mx-auto bg-white bg-opacity-10 backdrop-blur-lg rounded-xl shadow-xl p-6"
        >
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-white">
              Welcome back {username}!
            </h1>
            <p className="text-white text-opacity-80 mt-2">Manage your classroom with ease</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <motion.div
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className="bg-white bg-opacity-20 backdrop-blur-sm p-6 rounded-xl shadow-lg cursor-pointer text-white"
              onClick={handleCheckAnswerSheets}
            >
              <div className="flex flex-col items-center justify-center h-full">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <h2 className="text-xl font-semibold">Check Answer Sheets</h2>
                <p className="text-sm text-center mt-1">Upload and evaluate answer sheets</p>
              </div>
            </motion.div>

            <motion.div
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className={`${activeSection === "students" ? "bg-indigo-600" : "bg-white bg-opacity-20"} backdrop-blur-sm p-6 rounded-xl shadow-lg cursor-pointer text-white`}
              onClick={() => setActiveSection("students")}
            >
              <div className="flex flex-col items-center justify-center h-full">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
                <h2 className="text-xl font-semibold">Manage Students</h2>
                <p className="text-sm text-center mt-1">View and manage student data</p>
              </div>
            </motion.div>

            <motion.div
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className={`${activeSection === "tests" ? "bg-indigo-600" : "bg-white bg-opacity-20"} backdrop-blur-sm p-6 rounded-xl shadow-lg cursor-pointer text-white`}
              onClick={() => setActiveSection("tests")}
            >
              <div className="flex flex-col items-center justify-center h-full">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                <h2 className="text-xl font-semibold">Manage Tests</h2>
                <p className="text-sm text-center mt-1">View and manage test data</p>
              </div>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className={`${activeSection === "materials" ? "bg-indigo-600" : "bg-white bg-opacity-20"} backdrop-blur-sm p-6 rounded-xl shadow-lg cursor-pointer text-white`}
              onClick={() => setActiveSection("materials")}
            >
              <div className="flex flex-col items-center justify-center h-full">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
                <h2 className="text-xl font-semibold">Study Materials</h2>
                <p className="text-sm text-center mt-1">Access and manage study materials</p>
              </div>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className={`${activeSection === "lesson-plans" ? "bg-indigo-600" : "bg-white bg-opacity-20"} backdrop-blur-sm p-6 rounded-xl shadow-lg cursor-pointer text-white`}
              onClick={() => setActiveSection("lesson-plans")}
            >
              <div className="flex flex-col items-center justify-center h-full">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                <h2 className="text-xl font-semibold">Lesson Plans</h2>
                <p className="text-sm text-center mt-1">Create and organize teaching materials</p>
              </div>
            </motion.div>

            <motion.div
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className={`${activeSection === "assignment-generator" ? "bg-indigo-600" : "bg-white bg-opacity-20"} backdrop-blur-sm p-6 rounded-xl shadow-lg cursor-pointer text-white`}
              onClick={() => setActiveSection("assignment-generator")}
            >
              <div className="flex flex-col items-center justify-center h-full">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z" />
                </svg>
                <h2 className="text-xl font-semibold">Assignment Generator</h2>
                <p className="text-sm text-center mt-1">Structured content for your classroom</p>
              </div>
            </motion.div>
          </div>

          <div className="mb-6 relative">
            <input
              type="text"
              placeholder={`Search ${activeSection}...`}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full p-3 pl-10 rounded-xl border border-white border-opacity-30 bg-white bg-opacity-10 text-white placeholder-white placeholder-opacity-60 focus:outline-none focus:ring-2 focus:ring-white focus:ring-opacity-40"
            />
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 absolute left-3 top-3.5 text-white text-opacity-70" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>

          <AnimatePresence mode="wait">
            {activeSection === "students" ? (
              <motion.div
                key="students"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="bg-white bg-opacity-90 backdrop-blur-sm p-6 rounded-xl shadow-md"
              >
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-2xl font-bold text-gray-800">Student Data</h2>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={handleAddStudent}
                    className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-4 py-2 rounded-lg shadow flex items-center"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                    </svg>
                    Add Student
                  </motion.button>
                </div>

                <AnimatePresence>
                  {showAddStudentForm && (
                    <motion.div
                      initial={{ opacity: 0, y: -20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.3 }}
                      className="mb-6 bg-gray-50 p-6 rounded-lg"
                    >
                      <h3 className="text-lg font-semibold text-gray-800 mb-4">Add New Student</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Roll Number</label>
                          <input
                            type="text"
                            name="rollNumber"
                            placeholder="Enter roll number"
                            value={newStudent.rollNumber}
                            onChange={handleInputChange}
                            className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
                          <input
                            type="text"
                            name="name"
                            placeholder="Enter full name"
                            value={newStudent.name}
                            onChange={handleInputChange}
                            className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                          <input
                            type="email"
                            name="email"
                            placeholder="Enter email address"
                            value={newStudent.email}
                            onChange={handleInputChange}
                            className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                          />
                        </div>
                      </div>
                      <div className="flex space-x-4 mt-4">
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={handleSaveStudent}
                          disabled={isLoading}
                          className={`bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-6 py-2 rounded-lg shadow flex items-center ${isLoading ? 'opacity-70 cursor-not-allowed' : ''}`}
                        >
                          {isLoading ? (
                            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                          ) : (
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                          Save
                        </motion.button>
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={() => setShowAddStudentForm(false)}
                          disabled={isLoading}
                          className="bg-gray-300 text-gray-700 px-6 py-2 rounded-lg shadow"
                        >
                          Cancel
                        </motion.button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {getFilteredData().length > 0 ? (
                  <div className="overflow-x-auto rounded-lg shadow">
                    <table className="min-w-full bg-white">
                      <thead>
                        <tr className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white">
                          <th
                            className="px-6 py-3 text-left text-sm font-semibold cursor-pointer"
                            onClick={() => handleSort("rollNumber")}
                          >
                            <div className="flex items-center">
                              Roll Number
                              {sortBy.field === "rollNumber" && (
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sortBy.direction === "asc" ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
                                </svg>
                              )}
                            </div>
                          </th>
                          <th
                            className="px-6 py-3 text-left text-sm font-semibold cursor-pointer"
                            onClick={() => handleSort("name")}
                          >
                            <div className="flex items-center">
                              Name
                              {sortBy.field === "name" && (
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sortBy.direction === "asc" ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
                                </svg>
                              )}
                            </div>
                          </th>
                          <th
                            className="px-6 py-3 text-left text-sm font-semibold cursor-pointer"
                            onClick={() => handleSort("email")}
                          >
                            <div className="flex items-center">
                              Email
                              {sortBy.field === "email" && (
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sortBy.direction === "asc" ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
                                </svg>
                              )}
                            </div>
                          </th>
                          <th className="px-6 py-3 text-left text-sm font-semibold">Weak Topics</th>
                          <th className="px-6 py-3 text-left text-sm font-semibold">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(getFilteredData() as Student[]).map((student, index) => (
                          <motion.tr
                            key={index}
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.2, delay: index * 0.05 }}
                            className="border-b border-gray-200 hover:bg-gray-50"
                          >
                            <td className="px-6 py-4 text-sm text-gray-700">{student.rollNumber}</td>
                            <td className="px-6 py-4 text-sm text-gray-700">{student.name}</td>
                            <td className="px-6 py-4 text-sm text-gray-700">{student.email}</td>
                            <td className="px-6 py-4 text-sm text-gray-700">{student.weakTopics}</td>
                            <td className="px-6 py-4 text-sm">
                              <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => openEmailModal(student)}
                                className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-3 py-1 rounded-lg text-xs flex items-center"
                              >
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                </svg>
                                Send Email
                              </motion.button>
                            </td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="bg-gray-50 p-8 rounded-lg text-center"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                    </svg>
                    <p className="text-gray-600">No student data available. Add your first student to get started.</p>
                  </motion.div>
                )}
              </motion.div>
            ) : activeSection === "tests" ? (
              <motion.div
                key="tests"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="bg-white bg-opacity-90 backdrop-blur-sm p-6 rounded-xl shadow-md"
              >
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-2xl font-bold text-gray-800">Test Data</h2>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setShowAddTestForm(true)}
                    className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-4 py-2 rounded-lg shadow flex items-center"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                    </svg>
                    Add Test
                  </motion.button>
                </div>

                <AnimatePresence>
                  {showAddTestForm && (
                    <motion.div
                      initial={{ opacity: 0, y: -20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.3 }}
                      className="mb-6 bg-gray-50 p-6 rounded-lg"
                    >
                      <h3 className="text-lg font-semibold text-gray-800 mb-4">Add New Test</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Test Name</label>
                          <input
                            type="text"
                            name="name"
                            placeholder="Enter test name"
                            value={newTest.name}
                            onChange={handleTestInputChange}
                            className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Test Date</label>
                          <input
                            type="date"
                            name="date"
                            value={newTest.date}
                            onChange={handleTestInputChange}
                            className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Subject</label>
                          <input
                            type="text"
                            name="subject"
                            placeholder="Enter subject"
                            value={newTest.subject}
                            onChange={handleTestInputChange}
                            className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Total Marks</label>
                          <input
                            type="number"
                            name="totalMarks"
                            placeholder="Enter total marks"
                            value={newTest.totalMarks}
                            onChange={handleTestInputChange}
                            className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                          />
                        </div>
                      </div>
                      <div className="flex space-x-4 mt-4">
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={handleAddTest}
                          disabled={isLoading}
                          className={`bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-6 py-2 rounded-lg shadow flex items-center ${isLoading ? 'opacity-70 cursor-not-allowed' : ''}`}
                        >
                          {isLoading ? (
                            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                          ) : (
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                          Save
                        </motion.button>
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={() => setShowAddTestForm(false)}
                          disabled={isLoading}
                          className="bg-gray-300 text-gray-700 px-6 py-2 rounded-lg shadow"
                        >
                          Cancel
                        </motion.button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {getFilteredData().length > 0 ? (
                  <div className="overflow-x-auto rounded-lg shadow">
                    <table className="min-w-full bg-white">
                      <thead>
                        <tr className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white">
                          <th
                            className="px-6 py-3 text-left text-sm font-semibold cursor-pointer"
                            onClick={() => handleSort("name")}
                          >
                            <div className="flex items-center">
                              Test Name
                              {sortBy.field === "name" && (
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sortBy.direction === "asc" ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
                                </svg>
                              )}
                            </div>
                          </th>
                          <th
                            className="px-6 py-3 text-left text-sm font-semibold cursor-pointer"
                            onClick={() => handleSort("date")}
                          >
                            <div className="flex items-center">
                              Date
                              {sortBy.field === "date" && (
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sortBy.direction === "asc" ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
                                </svg>
                              )}
                            </div>
                          </th>
                          <th
                            className="px-6 py-3 text-left text-sm font-semibold cursor-pointer"
                            onClick={() => handleSort("subject")}
                          >
                            <div className="flex items-center">
                              Subject
                              {sortBy.field === "subject" && (
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sortBy.direction === "asc" ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
                                </svg>
                              )}
                            </div>
                          </th>
                          <th
                            className="px-6 py-3 text-left text-sm font-semibold cursor-pointer"
                            onClick={() => handleSort("status")}
                          >
                            <div className="flex items-center">
                              Status
                              {sortBy.field === "status" && (
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sortBy.direction === "asc" ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
                                </svg>
                              )}
                            </div>
                          </th>
                          <th
                            className="px-6 py-3 text-left text-sm font-semibold cursor-pointer"
                            onClick={() => handleSort("totalMarks")}
                          >
                            <div className="flex items-center">
                              Total Marks
                              {sortBy.field === "totalMarks" && (
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={sortBy.direction === "asc" ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"} />
                                </svg>
                              )}
                            </div>
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {(getFilteredData() as Test[]).map((test, index) => (
                          <motion.tr
                            key={index}
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.2, delay: index * 0.05 }}
                            className="border-b border-gray-200 hover:bg-gray-50"
                          >
                            <td className="px-6 py-4 text-sm text-gray-700">{test.name}</td>
                            <td className="px-6 py-4 text-sm text-gray-700">{new Date(test.date).toLocaleDateString()}</td>
                            <td className="px-6 py-4 text-sm text-gray-700">{test.subject}</td>
                            <td className="px-6 py-4 text-sm">
                              <span className={`px-3 py-1 rounded-full text-xs font-medium ${test.status === "Evaluated" ? "bg-green-100 text-green-800" : "bg-yellow-100 text-yellow-800"
                                }`}>
                                {test.status}
                              </span>
                            </td>
                            <td className="px-6 py-4 text-sm text-gray-700">{test.totalMarks}</td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="bg-gray-50 p-8 rounded-lg text-center"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                    <p className="text-gray-600">No test data available. Add your first test to get started.</p>
                  </motion.div>
                )}
              </motion.div>
            ) : activeSection === "materials" ? (
              <motion.div
                key="materials"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="bg-white bg-opacity-90 backdrop-blur-sm p-6 rounded-xl shadow-md"
              >
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-2xl font-bold text-gray-800">Study Materials</h2>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setShowUploadModal(true)}
                    className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-4 py-2 rounded-lg shadow flex items-center"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                    Upload File
                  </motion.button>
                </div>

                {books.length > 0 ? (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {books.map((book) => (
                      <motion.div
                        key={book.id}
                        whileHover={{ scale: 1.03 }}
                        whileTap={{ scale: 0.97 }}
                        className="bg-gray-50 p-4 rounded-lg cursor-pointer"
                        onClick={() => window.open(book.path, '_blank')}
                      >
                        <div className="text-center">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                          </svg>
                          <p className="mt-2 text-sm font-medium text-gray-700 truncate">{book.name}</p>
                          <p className="text-xs text-gray-500">Uploaded: {new Date(book.uploadedAt).toLocaleDateString()}</p>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="bg-gray-50 p-8 rounded-lg text-center"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                    </svg>
                    <p className="text-gray-600">Hmm... Where could I have kept the books?</p>
                  </motion.div>
                )}
              </motion.div>
            ) : activeSection === "lesson-plans" ? (
              <motion.div
                key="lesson-plans"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="bg-white bg-opacity-90 backdrop-blur-sm p-6 rounded-xl shadow-md"
              >
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-2xl font-bold text-gray-800">Lesson Plans</h2>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setShowLessonPlanForm(true)}
                    className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-4 py-2 rounded-lg shadow flex items-center"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                    </svg>
                    Create Plan
                  </motion.button>
                </div>

                {showLessonPlanForm && (
                  <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="mb-6 bg-gray-50 p-6 rounded-lg"
                  >
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Create New Lesson Plan</h3>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Plan Name</label>
                        <input
                          type="text"
                          name="name"
                          value={newLessonPlan.name}
                          onChange={handleLessonPlanInputChange}
                          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Subject</label>
                        <input
                          type="text"
                          name="subject"
                          value={newLessonPlan.subject}
                          onChange={handleLessonPlanInputChange}
                          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Topics (comma separated)</label>
                        <textarea
                          name="topics"
                          value={newLessonPlan.topics}
                          onChange={handleLessonPlanInputChange}
                          rows={3}
                          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Duration (hours): {newLessonPlan.duration}
                        </label>
                        <input
                          type="range"
                          name="duration"
                          min="1"
                          max="20"
                          value={newLessonPlan.duration}
                          onChange={handleLessonPlanInputChange}
                          className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer"
                        />
                      </div>
                    </div>
                    <div className="flex space-x-4 mt-4">
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={handleGenerateLessonPlan}
                        className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-6 py-2 rounded-lg shadow"
                      >
                        Generate Plan
                      </motion.button>
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setShowLessonPlanForm(false)}
                        className="bg-gray-300 text-gray-700 px-6 py-2 rounded-lg shadow"
                      >
                        Cancel
                      </motion.button>
                    </div>
                  </motion.div>
                )}

                {lessonPlans.length > 0 ? (
                  <div className="overflow-x-auto rounded-lg shadow">
                    <table className="min-w-full bg-white">
                      <thead>
                        <tr className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white">
                          <th className="px-6 py-3 text-left text-sm font-semibold">Name</th>
                          <th className="px-6 py-3 text-left text-sm font-semibold">Subject</th>
                          <th className="px-6 py-3 text-left text-sm font-semibold">Topics</th>
                          <th className="px-6 py-3 text-left text-sm font-semibold">Duration (hours)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {lessonPlans.map((plan, index) => (
                          <motion.tr
                            key={index}
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.2, delay: index * 0.05 }}
                            className="border-b border-gray-200 hover:bg-gray-50"
                          >
                            <td className="px-6 py-4 text-sm text-gray-700">{plan.name}</td>
                            <td className="px-6 py-4 text-sm text-gray-700">{plan.subject}</td>
                            <td className="px-6 py-4 text-sm text-gray-700">{plan.topics.join(', ')}</td>
                            <td className="px-6 py-4 text-sm text-gray-700">{plan.duration}</td>
                          </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="bg-gray-50 p-8 rounded-lg text-center"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mx-auto text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                    <p className="text-gray-600">No lesson plans created yet. Create your first lesson plan to get started.</p>
                  </motion.div>
                )}
              </motion.div>
            ) : activeSection === "assignment-generator" ? (
              <motion.div
                key="assignment-generator"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="bg-white bg-opacity-90 backdrop-blur-sm p-6 rounded-xl shadow-md"
              >
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-2xl font-bold text-gray-800">Assignment Generator</h2>
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setShowAssignmentGenerator(true)}
                    className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-4 py-2 rounded-lg shadow flex items-center"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                    </svg>
                    Generate Assignment
                  </motion.button>
                </div>

                {showAssignmentGenerator && (
                  <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="mb-6 bg-gray-50 p-6 rounded-lg"
                  >
                    <h3 className="text-lg font-semibold text-gray-800 mb-4">Assignment Options</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="md:col-span-2">
                        <label className="block text-sm font-medium text-gray-700 mb-1">Topics</label>
                        <textarea
                          name="topics"
                          value={assignmentOptions.topics}
                          onChange={handleAssignmentOptionChange}
                          rows={2}
                          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                          placeholder="Enter topics (comma separated)"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Number of Questions</label>
                        <input
                          type="number"
                          name="questionCount"
                          min="1"
                          max="50"
                          value={assignmentOptions.questionCount}
                          onChange={handleAssignmentOptionChange}
                          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Difficulty: {['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard'][assignmentOptions.difficulty - 1]}
                        </label>
                        <input
                          type="range"
                          name="difficulty"
                          min="1"
                          max="5"
                          value={assignmentOptions.difficulty}
                          onChange={handleAssignmentOptionChange}
                          className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer"
                        />
                      </div>

                      <div className="md:col-span-2">
                        <label className="block text-sm font-medium text-gray-700 mb-1">Type</label>
                        <div className="flex space-x-4">
                          {['HW', 'Test', 'Assignment', 'Custom'].map(type => (
                            <label key={type} className="inline-flex items-center">
                              <input
                                type="radio"
                                name="type"
                                value={type}
                                checked={assignmentOptions.type === type}
                                onChange={handleAssignmentOptionChange}
                                className="text-indigo-600 focus:ring-indigo-500"
                              />
                              <span className="ml-2">{type}</span>
                            </label>
                          ))}
                        </div>
                        {assignmentOptions.type === 'Custom' && (
                          <input
                            type="text"
                            name="customType"
                            value={assignmentOptions.customType}
                            onChange={handleAssignmentOptionChange}
                            placeholder="Enter custom type"
                            className="w-full mt-2 p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                          />
                        )}
                      </div>

                      <div className="md:col-span-2">
                        <label className="block text-sm font-medium text-gray-700 mb-1">Question Types</label>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                          {['MCQ', 'Short Answer', 'Long Answer', 'Fill in the Blanks', 'True/False', 'Matching'].map(type => (
                            <label key={type} className="inline-flex items-center">
                              <input
                                type="checkbox"
                                name="questionTypes"
                                value={type}
                                checked={assignmentOptions.questionTypes.includes(type)}
                                onChange={handleAssignmentOptionChange}
                                className="text-indigo-600 focus:ring-indigo-500"
                              />
                              <span className="ml-2">{type}</span>
                            </label>
                          ))}
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Total Marks</label>
                        <input
                          type="number"
                          name="totalMarks"
                          min="1"
                          value={assignmentOptions.totalMarks}
                          onChange={handleAssignmentOptionChange}
                          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Marks Distribution</label>
                        <select
                          name="marksDistribution"
                          value={assignmentOptions.marksDistribution}
                          onChange={handleAssignmentOptionChange}
                          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                        >
                          <option value="equal">Equal per question</option>
                          <option value="smart">Smart distribution</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Duration (minutes)</label>
                        <input
                          type="number"
                          name="duration"
                          min="1"
                          value={assignmentOptions.duration}
                          onChange={handleAssignmentOptionChange}
                          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Bloom's Level</label>
                        <select
                          name="bloomLevel"
                          value={assignmentOptions.bloomLevel}
                          onChange={handleAssignmentOptionChange}
                          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                        >
                          <option value="Remembering">Remembering</option>
                          <option value="Understanding">Understanding</option>
                          <option value="Applying">Applying</option>
                          <option value="Analyzing">Analyzing</option>
                          <option value="Evaluating">Evaluating</option>
                          <option value="Creating">Creating</option>
                        </select>
                      </div>

                      <div className="md:col-span-2">
                        <label className="inline-flex items-center">
                          <input
                            type="checkbox"
                            name="includeAnswerKey"
                            checked={assignmentOptions.includeAnswerKey}
                            onChange={handleAssignmentOptionChange}
                            className="text-indigo-600 focus:ring-indigo-500"
                          />
                          <span className="ml-2">Include Answer Key</span>
                        </label>
                      </div>

                      <div className="md:col-span-2">
                        <label className="block text-sm font-medium text-gray-700 mb-1">Instructions</label>
                        <textarea
                          name="instructions"
                          value={assignmentOptions.instructions}
                          onChange={handleAssignmentOptionChange}
                          rows={2}
                          className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                          placeholder="Optional instructions for students"
                        />
                      </div>
                    </div>

                    <div className="flex space-x-4 mt-4">
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={handleGenerateAssignment}
                        className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-6 py-2 rounded-lg shadow"
                      >
                        Generate Assignment
                      </motion.button>
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setShowAssignmentGenerator(false)}
                        className="bg-gray-300 text-gray-700 px-6 py-2 rounded-lg shadow"
                      >
                        Cancel
                      </motion.button>
                    </div>
                  </motion.div>
                )}
              </motion.div>
            ) : null}
          </AnimatePresence>

          <AnimatePresence>
            {showEmailModal && selectedEmailStudent && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
              >
                <motion.div
                  initial={{ scale: 0.9, y: 20 }}
                  animate={{ scale: 1, y: 0 }}
                  exit={{ scale: 0.9, y: 20 }}
                  className="bg-white rounded-xl shadow-xl p-6 w-full max-w-md"
                >
                  <h3 className="text-xl font-bold text-gray-800 mb-4">
                    Send Email to {selectedEmailStudent.name}
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">
                    {selectedEmailStudent.email}
                  </p>

                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Email Subject
                    </label>
                    <textarea
                      rows={5}
                      value={emailContent}
                      onChange={(e) => setEmailContent(e.target.value)}
                      placeholder="Enter your message here..."
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    ></textarea>
                  </div>

                  <div className="flex space-x-4">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={handleSendEmail}
                      disabled={isLoading}
                      className={`bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-6 py-2 rounded-lg shadow flex items-center ${isLoading ? 'opacity-70 cursor-not-allowed' : ''}`}
                    >
                      {isLoading ? (
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                      ) : (
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                      )}
                      Send Email
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setShowEmailModal(false)}
                      disabled={isLoading}
                      className="bg-gray-300 text-gray-700 px-6 py-2 rounded-lg shadow"
                    >
                      Cancel
                    </motion.button>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
          <AnimatePresence>
            {showUploadModal && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
              >
                <motion.div
                  initial={{ scale: 0.9, y: 20 }}
                  animate={{ scale: 1, y: 0 }}
                  exit={{ scale: 0.9, y: 20 }}
                  className="bg-white rounded-xl shadow-xl p-6 w-full max-w-md"
                >
                  <h3 className="text-xl font-bold text-gray-800 mb-4">Upload Study Material</h3>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                    <input
                      type="file"
                      accept="application/pdf"
                      onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                      className="hidden"
                      id="materialFileInput"
                    />
                    <label htmlFor="materialFileInput" className="cursor-pointer">
                      {selectedFile ? (
                        <p className="text-sm text-gray-700">{selectedFile.name}</p>
                      ) : (
                        <>
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-8 w-8 mx-auto text-gray-400"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                            />
                          </svg>
                          <p className="mt-2 text-sm text-gray-600">Click to upload PDF</p>
                        </>
                      )}
                    </label>
                  </div>
                  <div className="flex space-x-4 mt-6">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={handleUpload}
                      disabled={isLoading || !selectedFile}
                      className={`bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-6 py-2 rounded-lg shadow flex items-center ${isLoading || !selectedFile ? "opacity-70 cursor-not-allowed" : ""
                        }`}
                    >
                      {isLoading ? (
                        <svg
                          className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                          ></circle>
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          ></path>
                        </svg>
                      ) : (
                        "Upload"
                      )}
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setShowUploadModal(false)}
                      className="bg-gray-300 text-gray-700 px-6 py-2 rounded-lg shadow"
                    >
                      Cancel
                    </motion.button>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
          <AnimatePresence>
            {showContentModal && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
              >
                <motion.div
                  initial={{ scale: 0.9, y: 20 }}
                  animate={{ scale: 1, y: 0 }}
                  exit={{ scale: 0.9, y: 20 }}
                  className="bg-white rounded-xl shadow-xl p-6 w-full max-w-4xl max-h-[80vh] flex flex-col"
                >
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-xl font-bold text-gray-800">Generated Content</h3>
                    <button onClick={() => setShowContentModal(false)} className="text-gray-500 hover:text-gray-700">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>

                  <div className="flex-1 overflow-auto bg-gray-50 p-4 rounded-lg mb-4">
                    <pre className="whitespace-pre-wrap font-sans text-sm">{generatedContent}</pre>
                  </div>

                  <div className="flex justify-end">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => {
                        navigator.clipboard.writeText(generatedContent);
                        showSuccessNotification("Content copied to clipboard!");
                      }}
                      className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white px-6 py-2 rounded-lg shadow"
                    >
                      Copy to Clipboard
                    </motion.button>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

      </div>

    </ProtectedRoute>
  );
};

export default Dashboard;
