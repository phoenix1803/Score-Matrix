import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
const studentsFilePath = path.join(process.cwd(), 'data', 'students.json');

interface Student {
  rollNumber: string;
  name: string;
  email: string;
  weakTopics: string;
}

export async function POST(request: Request) {
  try {
    const studentsData = fs.readFileSync(studentsFilePath, 'utf8');
    const students: Student[] = JSON.parse(studentsData);

    const newStudent: Student = await request.json();

    students.push(newStudent);

    fs.writeFileSync(studentsFilePath, JSON.stringify(students, null, 2));

    return NextResponse.json(
      { message: 'Student added successfully!', student: newStudent },
      { status: 201 }
    );
  } catch (error) {
    console.error('Error saving student:', error);
    return NextResponse.json(
      { message: 'Failed to add student' },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const studentsData = fs.readFileSync(studentsFilePath, 'utf8');
    const students: Student[] = JSON.parse(studentsData);

    return NextResponse.json(students, { status: 200 });
  } catch (error) {
    console.error('Error fetching students:', error);
    return NextResponse.json(
      { message: 'Failed to fetch students' },
      { status: 500 }
    );
  }
}