import google.generativeai as genai
genai.configure(api_key="AIzaSyAQvW-7i3jnNu5qwolDOPV9q2HhdkKtrAU")
def generate_test_questions(text, num_questions, difficulty_level):
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    if num_questions <= 10:
        question_types = ["multiple_choice"] * num_questions
    else:
        question_types = []
        question_types += ["multiple_choice"] * (num_questions // 2)
        question_types += ["2_marker"] * (num_questions // 5)
        question_types += ["3_marker"] * (num_questions // 5)
        question_types += ["fill_in_the_blank"] * (num_questions - len(question_types))
    prompt = f"""
    Generate a test based on the following text(make sure they are in normal formatting):
    {text}
    The test should have {num_questions} questions of {difficulty_level} difficulty level.
    Include a variety of question types: multiple-choice, 2-markers, 3-markers, and fill-in-the-blanks.
    Follow these guidelines for each question type:
    1. Multiple-Choice Questions:
       - Provide 4 options.
       - Clearly indicate the correct answer.
       Example:
       Question 1: What is the capital of France?
       A) London
       B) Paris
       C) Berlin
       D) Madrid
       Correct Answer: B) Paris
    
    2. 2-Marker Questions:
       - Ask a short-answer question worth 2 marks.
       - Provide a concise correct answer.(30 words)
       Example:
       Question 2: Name the process by which plants make their food.
       Correct Answer: Photosynthesis
    
    3. 3-Marker Questions:
       - Ask a question worth 3 marks that requires a brief explanation or reasoning.
       - Provide a clear and concise correct answer.(50 words)
       Example:
       Question 3: Explain why the sky appears blue.
       Correct Answer: The sky appears blue due to Rayleigh scattering, where shorter blue wavelengths of sunlight are scattered in all directions by the gases and particles in the Earth's atmosphere.
    
    4. Fill-in-the-Blank Questions:
       - Provide a sentence with a blank and the correct word to fill in.
       Example:
       Question 4: The process of water turning into vapor is called __________.
       Correct Answer: evaporation
    
    Generate the test questions now:
    """

    response = model.generate_content(prompt)
    return response.text
def main():
    text = "Test on applied chemistry , BTech" ##here change
    num_questions = "12" ##here change
    difficulty_level = "hard" ##here change
    if difficulty_level not in ["easy", "medium", "hard"]:
        print("Invalid difficulty level. Please choose from 'easy', 'medium', or 'hard'.")
        return
    test_questions = generate_test_questions(text, num_questions, difficulty_level)
    print("\nGenerated Test Questions:\n")
    print(test_questions)

if __name__ == "__main__":
    main()