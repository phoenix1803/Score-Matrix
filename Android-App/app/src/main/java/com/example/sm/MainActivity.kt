package com.example.sm




import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.platform.LocalClipboardManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.content

import android.app.Activity.RESULT_OK
import androidx.compose.runtime.Composable
import android.content.ContentValues
import android.content.Context
import android.graphics.pdf.PdfDocument
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent

import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource

import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController

import androidx.navigation.compose.*
import com.example.scorematrix.GeminiEvaluator
import com.google.mlkit.vision.documentscanner.*
import dev.jeziellago.compose.markdowntext.MarkdownText
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

import java.io.IOException
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.keyframes
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.animation.core.tween
import kotlinx.coroutines.delay
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Search
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.room.Room
import androidx.room.parser.Section.Companion.text
import com.example.sm.ui.theme.StudentDataScreen


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            ScoreMatrixApp(this)
        }
    }
}

@Composable
fun ScoreMatrixApp(activity: ComponentActivity) {
    val navController = rememberNavController()
    val studentState = remember { StudentState() }
    val evalState: EvaluationState = viewModel()

    NavHost(navController, startDestination = "main") {
        composable("main") { MainScreen(activity, navController) }
        composable("evaluation") { EvaluationScreen(navController, evalState) }  // <-- pass evalState here
        composable("students") { StudentDataScreen(navController) }
        composable("addStudent") { AddStudentScreen(navController, studentState) }
        composable("paperCheckerOptions") { PaperCheckerOptionsScreen(activity, navController) }
        composable("parameterSliders") {
            ParameterSliderScreen(
                navController = navController,
                evalState = evalState
            )
        }
        composable("generate_assignment") {
            AssignmentGeneratorScreen(navController = navController)
        }
        composable("lesson_plan") {
            LessonPlanScreen(navController = navController)
        }

    }
}
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PaperCheckerOptionsScreen(
    activity: ComponentActivity,
    navController: NavController
) {
    val context = LocalContext.current

    val scannerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartIntentSenderForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK && result.data != null) {
            val scanResult = GmsDocumentScanningResult.fromActivityResultIntent(result.data)
            scanResult?.let { result ->
                val imageUris = result.pages?.map { it.imageUri } ?: emptyList()
                if (imageUris.isNotEmpty()) {
                    saveImagesAsPdf(context, imageUris)
                } else {
                    Toast.makeText(context, "No images were scanned", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.DarkGray)
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = {
                        Text("Paper Checker", color = Color.Black)
                    },
                    navigationIcon = {
                        IconButton(onClick = { navController.popBackStack() }) {
                            Icon(
                                imageVector = Icons.Default.ArrowBack,
                                contentDescription = "Back",
                                tint = Color.White
                            )
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = Color(0xFF5D4A79)
                    )
                )
            },
            containerColor = Color(0xFFFDFDFF)
        ) { innerPadding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(24.dp, Alignment.Top),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Spacer(modifier = Modifier.height(32.dp))

                // Scan Documents Button
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(180.dp)
                        .border(3.dp, MaterialTheme.colorScheme.onSurface, RoundedCornerShape(29.dp))
                        .clickable { startScanning(activity, scannerLauncher) },
                    shape = RoundedCornerShape(29.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(MaterialTheme.colorScheme.inversePrimary)
                            .padding(horizontal = 16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Image(
                            painter = painterResource(id = R.drawable.scan),
                            contentDescription = "Scan Icon",
                            modifier = Modifier.size(100.dp)
                        )
                        Text(
                            text = "Scan Documents",
                            fontSize = 20.sp,
                            color = MaterialTheme.colorScheme.onSurface,
                            textDecoration = TextDecoration.Underline
                        )
                    }
                }

                // Evaluate Documents Button
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(180.dp)
                        .border(3.dp, MaterialTheme.colorScheme.onSurface, RoundedCornerShape(29.dp))
                        .clickable { navController.navigate("evaluation") },
                    shape = RoundedCornerShape(29.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(MaterialTheme.colorScheme.inversePrimary)
                            .padding(horizontal = 16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Image(
                            painter = painterResource(id = R.drawable.evaluate),
                            contentDescription = "Evaluate Icon",
                            modifier = Modifier.size(100.dp)
                        )
                        Text(
                            text = "Evaluate Documents",
                            fontSize = 20.sp,
                            color = MaterialTheme.colorScheme.onSurface,
                            textDecoration = TextDecoration.Underline
                        )
                    }
                }
            }
        }
    }
}


data class Student(
    val rollNumber: String,
    val name: String,
    val email: String = ""
)

// 2. StudentItem composable
@Composable
fun StudentItem(student: Student) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Roll Number: ${student.rollNumber}",
                style = MaterialTheme.typography.bodyLarge)
            Text("Name: ${student.name}",
                style = MaterialTheme.typography.bodyMedium)
            if (student.email.isNotBlank()) {
                Text("Email: ${student.email}",
                    style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(activity: ComponentActivity, navController: NavController) {
    var imageUris by remember { mutableStateOf<List<Uri>>(emptyList()) }
    val context = LocalContext.current

    val scannerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartIntentSenderForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK && result.data != null) {
            val scanResult = GmsDocumentScanningResult.fromActivityResultIntent(result.data)
            scanResult?.let { result ->
                imageUris = result.pages?.map { it.imageUri } ?: emptyList()
                if (imageUris.isNotEmpty()) {
                    saveImagesAsPdf(context, imageUris)
                }
            }
        }
    }

    val pdfPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            Toast.makeText(context, "Book uploaded successfully", Toast.LENGTH_SHORT).show()
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.DarkGray)
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = {
                        Image(
                            painter = painterResource(id = R.drawable.logo),
                            contentDescription = "App Logo",
                            modifier = Modifier
                                .fillMaxWidth()
                                .wrapContentSize(Alignment.Center)
                                .size(450.dp)
                                .offset(x = (-8).dp),
                            contentScale = ContentScale.Fit
                        )
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = Color(0xFF5D4A79)
                    )
                )
            },
            containerColor = Color(0xFFFDFDFF)
        ) { innerPadding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Spacer(modifier = Modifier.height(45.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(20.dp)
                ) {
                    // 1st Card: Add Student
                    Card(
                        modifier = Modifier
                            .weight(1f)
                            .aspectRatio(1f)
                            .border(3.dp, MaterialTheme.colorScheme.onSurface, shape = RoundedCornerShape(29.dp))
                            .clickable { navController.navigate("addStudent") },
                        shape = RoundedCornerShape(29.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(MaterialTheme.colorScheme.inversePrimary),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center
                        ) {
                            Image(
                                painter = painterResource(id = R.drawable.add_student),
                                contentDescription = "add student",
                                modifier = Modifier.size(100.dp)
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Add Student",
                                fontSize = 18.sp,
                                color = MaterialTheme.colorScheme.onSurface,
                                textDecoration = TextDecoration.Underline,
                                modifier = Modifier.padding(1.dp)
                            )
                        }
                    }

                    // 2nd Card: Paper Checker
                    Card(
                        modifier = Modifier
                            .weight(1f)
                            .aspectRatio(1f)
                            .border(3.dp, MaterialTheme.colorScheme.onSurface, shape = RoundedCornerShape(29.dp))
                            .clickable { navController.navigate("paperCheckerOptions") },
                        shape = RoundedCornerShape(29.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(MaterialTheme.colorScheme.inversePrimary),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center
                        ) {
                            Image(
                                painter = painterResource(id = R.drawable.evaluate),
                                contentDescription = "Evaluate Paper",
                                modifier = Modifier.size(100.dp)
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Evaluate Paper",
                                fontSize = 17.sp,
                                color = MaterialTheme.colorScheme.onSurface,
                                textDecoration = TextDecoration.Underline,
                                modifier = Modifier.padding(1.dp)
                            )
                        }
                    }
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(20.dp)
                ) {
                    // 3rd Card: Upload Books
                    Card(
                        modifier = Modifier
                            .weight(1f)
                            .aspectRatio(1f)
                            .border(3.dp, MaterialTheme.colorScheme.onSurface, shape = RoundedCornerShape(29.dp))
                            .clickable {
                                pdfPickerLauncher.launch("application/pdf")
                            },
                        shape = RoundedCornerShape(29.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(MaterialTheme.colorScheme.inversePrimary),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center
                        ) {
                            Image(
                                painter = painterResource(id = R.drawable.upload),
                                contentDescription = "Results",
                                modifier = Modifier.size(100.dp)
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Upload Books",
                                fontSize = 17.sp,
                                color = MaterialTheme.colorScheme.onSurface,
                                textDecoration = TextDecoration.Underline,
                                modifier = Modifier.padding(1.dp)
                            )
                        }
                    }

                    // 4th Card: manage test
                    Card(
                        modifier = Modifier
                            .weight(1f)
                            .aspectRatio(1f)
                            .border(3.dp, MaterialTheme.colorScheme.onSurface, shape = RoundedCornerShape(29.dp))
                            .clickable {
                                Toast.makeText(context, "Coming soon!", Toast.LENGTH_SHORT).show()
                            },
                        shape = RoundedCornerShape(29.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(MaterialTheme.colorScheme.inversePrimary),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center
                        ) {
                            Image(
                                painter = painterResource(id = R.drawable.manage),
                                contentDescription = "Settings",
                                modifier = Modifier.size(110.dp)
                            )
                            Spacer(modifier = Modifier.height(1.dp))
                            Text(
                                text = "Manage Tests",
                                fontSize = 17.9.sp,
                                color = MaterialTheme.colorScheme.onSurface,
                                textDecoration = TextDecoration.Underline,
                                modifier = Modifier.padding(1.dp)
                            )
                        }
                    }
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(20.dp)
                ) {
                    // 5th Card: Lesson Planner
                    Card(
                        modifier = Modifier
                            .weight(1f)
                            .aspectRatio(1f)
                            .border(3.dp, MaterialTheme.colorScheme.onSurface, shape = RoundedCornerShape(29.dp))
                            .clickable { navController.navigate("lesson_plan") },
                        shape = RoundedCornerShape(29.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(MaterialTheme.colorScheme.inversePrimary),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center
                        ) {
                            Image(
                                painter = painterResource(id = R.drawable.lesson_plan),
                                contentDescription = "Results",
                                modifier = Modifier.size(100.dp)
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Lesson Planner",
                                fontSize = 17.sp,
                                color = MaterialTheme.colorScheme.onSurface,
                                textDecoration = TextDecoration.Underline,
                                modifier = Modifier.padding(1.dp)
                            )
                        }
                    }

                    // 6th Card: Generate Assignment
                    Card(
                        modifier = Modifier
                            .weight(1f)
                            .aspectRatio(1f)
                            .border(3.dp, MaterialTheme.colorScheme.onSurface, shape = RoundedCornerShape(29.dp))
                            .clickable { navController.navigate("generate_assignment")
                                 },
                        shape = RoundedCornerShape(29.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(MaterialTheme.colorScheme.inversePrimary),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center
                        ) {
                            Image(
                                painter = painterResource(id = R.drawable.generate),
                                contentDescription = "Settings",
                                modifier = Modifier.size(110.dp)
                            )
                            Spacer(modifier = Modifier.height(1.dp))
                            Text(
                                text = "Generate Assignment",
                                fontSize = 17.9.sp,
                                color = MaterialTheme.colorScheme.onSurface,
                                textDecoration = TextDecoration.Underline,
                                modifier = Modifier.padding(1.dp)
                            )
                        }
                    }
                }
            }
        }
    }
}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AssignmentGeneratorScreen(
    navController: NavController,
    viewModel: AssignmentViewModel = viewModel()
) {
    val state by viewModel.uiState.collectAsState()
    val context = LocalContext.current

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Generate Assignment") },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary
                ),
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Back",
                            tint = Color.White
                        )
                    }
                }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .padding(innerPadding)
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {

            OutlinedTextField(
                value = state.topics,
                onValueChange = viewModel::onTopicsChange,
                label = { Text("Topics of Test") },
                modifier = Modifier.fillMaxWidth()
            )

            OutlinedTextField(
                value = state.numQuestions,
                onValueChange = viewModel::onNumQuestionsChange,
                label = { Text("Number of Questions") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )

            Text("Difficulty")
            Slider(
                value = state.difficulty.toFloat(),
                onValueChange = { viewModel.onDifficultyChange(it.toInt()) },
                valueRange = 1f..5f,
                steps = 3
            )

            AssignmentTypeSelector(
                selectedType = state.assignmentType,
                onTypeChange = viewModel::onAssignmentTypeChange,
                customText = state.customType
            )

            QuestionTypeMultiSelect(
                selectedTypes = state.questionTypes,
                onTypeToggle = viewModel::onQuestionTypeToggle
            )

            TotalMarksSection(state, viewModel)

            OutlinedTextField(
                value = state.duration,
                onValueChange = viewModel::onDurationChange,
                label = { Text("Total Duration (mins)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                modifier = Modifier.fillMaxWidth()
            )

            BloomLevelDropdown(
                selected = state.bloomLevel,
                onChange = viewModel::onBloomLevelChange
            )

            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Include Answer Key?")
                Spacer(Modifier.width(8.dp))
                Switch(
                    checked = state.includeAnswers,
                    onCheckedChange = viewModel::onIncludeAnswersToggle
                )
            }

            OutlinedTextField(
                value = state.instructions,
                onValueChange = viewModel::onInstructionsChange,
                label = { Text("Instructions / Notes") },
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(20.dp))
            val isLoading = state.isLoading


            Button(
                onClick = { viewModel.generateAssignment(context) },
                modifier = Modifier.fillMaxWidth(),
                enabled = !isLoading
            ) {
                if (isLoading) {
                    CircularProgressIndicator(
                        color = Color.White,
                        modifier = Modifier.size(20.dp)
                    )
                } else {
                    Text("Generate Assignment")
                }
            }
        }

            if (state.generatedAssignment.isNotEmpty()) {
            GeneratedAssignmentDialog(
                text = state.generatedAssignment,
                onDismiss = { viewModel.clearGeneratedText() }
            )
        }
    }
}
@Composable
fun GeneratedAssignmentDialog(
    titleText: String = "Generated Assignment",
    text: String,
    onDismiss: () -> Unit
) {
    val clipboard = LocalClipboardManager.current
    val paragraphs = remember(text) { text.split("\n\n").filter { it.isNotBlank() } }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(titleText) },
        text = {
            Column {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 400.dp)
                ) {
                    items(paragraphs) { paragraph ->
                        MarkdownText(
                            markdown = paragraph,
                            modifier = Modifier.padding(vertical = 4.dp)
                        )
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                Button(
                    onClick = { clipboard.setText(AnnotatedString(text)) },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Copy to Clipboard")
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Close")
            }
        }
    )
}

data class AssignmentUIState(
    val topics: String = "",
    val isLoading: Boolean = false,
    val numQuestions: String = "",
    val difficulty: Int = 3,
    val assignmentType: String = "Homework",
    val customType: String = "",
    val questionTypes: Set<String> = emptySet(),
    val totalMarks: String = "",
    val equalDistribution: Boolean = true,
    val duration: String = "",
    val bloomLevel: String = "",
    val includeAnswers: Boolean = true,
    val instructions: String = "",
    val generatedAssignment: String = ""
)

class AssignmentViewModel : ViewModel() {
    private val _uiState = MutableStateFlow(AssignmentUIState())
    val uiState: StateFlow<AssignmentUIState> = _uiState

    fun onTopicsChange(value: String) { _uiState.update { it.copy(topics = value) } }
    fun onNumQuestionsChange(value: String) { _uiState.update { it.copy(numQuestions = value) } }
    fun onDifficultyChange(value: Int) { _uiState.update { it.copy(difficulty = value) } }
    fun onAssignmentTypeChange(value: String) {
        _uiState.update { it.copy(assignmentType = value, customType = if (value != "Custom") "" else it.customType) }
    }
    fun onTotalMarksChange(value: String) {
        _uiState.update { it.copy(totalMarks = value) }
    }

    fun onEqualDistributionChange(equal: Boolean) {
        _uiState.update { it.copy(equalDistribution = equal) }
    }

    fun onQuestionTypeToggle(type: String) {
        _uiState.update {
            val newSet = it.questionTypes.toMutableSet()
            if (newSet.contains(type)) newSet.remove(type) else newSet.add(type)
            it.copy(questionTypes = newSet)
        }
    }
    fun onDurationChange(value: String) { _uiState.update { it.copy(duration = value) } }
    fun onBloomLevelChange(value: String) { _uiState.update { it.copy(bloomLevel = value) } }
    fun onIncludeAnswersToggle(value: Boolean) { _uiState.update { it.copy(includeAnswers = value) } }
    fun onInstructionsChange(value: String) { _uiState.update { it.copy(instructions = value) } }
    fun clearGeneratedText() { _uiState.update { it.copy(generatedAssignment = "") } }

    fun generateAssignment(context: Context) {
        _uiState.update { it.copy(isLoading = true) }

        viewModelScope.launch {
            val prompt = buildAssignmentPrompt(_uiState.value)
            val response = callGemini(prompt, context)
            _uiState.update {
                it.copy(
                    generatedAssignment = response,
                    isLoading = false
                )
            }
        }
    }


    private fun buildAssignmentPrompt(state: AssignmentUIState): String {
        return """
        You are a strict teacher generating an assignment.

        Your task is to generate exactly **${state.numQuestions}** questions for a "${state.assignmentType.takeIf { it != "Custom" } ?: state.customType}" on the topic "${state.topics}".

        Question Types: ${state.questionTypes.joinToString(", ")}
        Difficulty: ${state.difficulty}/5
        Total Marks: ${state.totalMarks}
        Distribution: ${if (state.equalDistribution) "Equal" else "Smart"}
        Duration: ${state.duration} minutes
        Bloom's Level: ${state.bloomLevel}
        ${if (state.includeAnswers) "Include a clear answer after each question." else "Do NOT include answers."}
        Instructions to include: ${state.instructions}

         Output format:
        1. Question 1
        2. Question 2
        ...
        Do not use emojis   
        
        ${state.numQuestions}. Question ${state.numQuestions}

        Do NOT skip any number.
        Generate exactly ${state.numQuestions} questions – no more, no less.

        If question types are mixed, randomly mix them across the paper.
    """.trimIndent()
    }



    private suspend fun callGemini(prompt: String, context: Context): String {
        val model = GenerativeModel(
            modelName = "gemini-1.5-flash",
            apiKey = "xxxxxxxxxxxxx"
        )
        return try {
            val response = model.generateContent(content { text(prompt) })
            response.text ?: "No output"
        } catch (e: Exception) {
            Log.e("AssignmentGen", "Error: ${e.localizedMessage}")
            "Failed: ${e.localizedMessage}"
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun QuestionTypeMultiSelect(selectedTypes: Set<String>, onTypeToggle: (String) -> Unit) {
    val types = listOf("MCQ", "Short Answer", "Long Answer", "Fill in the Blanks")
    Column {
        Text("Question Types")
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            types.forEach { type ->
                FilterChip(
                    selected = selectedTypes.contains(type),
                    onClick = { onTypeToggle(type) },
                    label = { Text(type) }
                )
            }
        }
    }
}

@Composable
fun TotalMarksSection(state: AssignmentUIState, viewModel: AssignmentViewModel) {
    OutlinedTextField(
        value = state.totalMarks,
        onValueChange = {viewModel.onTotalMarksChange(it) },
        label = { Text("Total Marks") },
        modifier = Modifier.fillMaxWidth()
    )
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(24.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            RadioButton(
                selected = state.equalDistribution,
                onClick = { viewModel.onEqualDistributionChange(true) }
            )
            Text("Equal")
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            RadioButton(
                selected = !state.equalDistribution,
                onClick = { viewModel.onEqualDistributionChange(false) }
            )
            Text("Smart")
        }
    }

}

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun BloomLevelDropdown(selected: String, onChange: (String) -> Unit) {
    val levels = listOf("Remembering", "Understanding", "Applying", "Analyzing", "Evaluating", "Creating")
    var expanded by remember { mutableStateOf(false) }

    Box {
        OutlinedTextField(
            value = selected,
            onValueChange = {},
            modifier = Modifier
                .fillMaxWidth()
                .combinedClickable(
                    onClick = { expanded = true }
                ),

            enabled = false,
            label = { Text("Bloom Level (optional)") }
        )
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            levels.forEach {
                DropdownMenuItem(
                    text = { Text(it) },
                    onClick = {
                        onChange(it)
                        expanded = false
                    }
                )
            }
        }
    }
}








@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EvaluationScreen(navController: NavController, evalState: EvaluationState) {
    val context = LocalContext.current

    val pickQuestionPaperLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        evalState.questionPaperUri = uri
    }

    val pickAnswerSheetLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        evalState.answerSheetUri = uri
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Answer Sheet Evaluation") },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary
                ),
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Back",
                            tint = Color.White
                        )
                    }
                }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(24.dp),
            verticalArrangement = Arrangement.Top,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Evaluate Answer Sheet",
                style = MaterialTheme.typography.headlineMedium,
                modifier = Modifier.padding(bottom = 32.dp)
            )

            Button(
                onClick = { pickQuestionPaperLauncher.launch("application/pdf") },
                modifier = Modifier
                    .fillMaxWidth(0.8f)
                    .padding(vertical = 8.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (evalState.questionPaperUri != null)
                        Color(0xFF4CAF50)
                    else
                        MaterialTheme.colorScheme.primary
                )
            ) {
                Text("Select Question Paper")
            }

            evalState.questionPaperUri?.let {
                Text(
                    text = "Selected: ${it.getFileName(context)}",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
            }

            Button(
                onClick = { pickAnswerSheetLauncher.launch("application/pdf") },
                modifier = Modifier
                    .fillMaxWidth(0.8f)
                    .padding(vertical = 8.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (evalState.answerSheetUri != null)
                        Color(0xFF4CAF50)
                    else
                        MaterialTheme.colorScheme.primary
                )
            ) {
                Text("Select Answer Sheet")
            }

            evalState.answerSheetUri?.let {
                Text(
                    text = "Selected: ${it.getFileName(context)}",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
            }

            Button(
                onClick = { navController.navigate("parameterSliders") },
                modifier = Modifier
                    .fillMaxWidth(0.8f)
                    .padding(vertical = 8.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Text("Set Evaluation Parameters")
            }

            Spacer(modifier = Modifier.height(24.dp))

            Button(
                onClick = {
                    if (evalState.questionPaperUri != null && evalState.answerSheetUri != null) {
                        evalState.isLoading = true
                        val evaluator = GeminiEvaluator(context)

                        CoroutineScope(Dispatchers.IO).launch {
                            val result = evaluator.evaluatePdfs(
                                evalState.questionPaperUri!!,
                                evalState.answerSheetUri!!,
                                evalState.sliderValues
                            )

                            withContext(Dispatchers.Main) {
                                evalState.isLoading = false
                                evalState.evaluationResult = result
                                Toast.makeText(context, "Evaluation complete!", Toast.LENGTH_SHORT).show()
                            }
                        }
                    } else {
                        Toast.makeText(context, "Please select both PDFs", Toast.LENGTH_SHORT).show()
                    }
                },
                modifier = Modifier.fillMaxWidth(0.8f),
                enabled = !evalState.isLoading
            ) {
                if (evalState.isLoading) {
                    CircularProgressIndicator(color = Color.White, modifier = Modifier.size(20.dp))
                } else {
                    Text("Evaluate")
                }
            }

            evalState.evaluationResult?.let {
                Spacer(modifier = Modifier.height(24.dp))
                GeminiMarkdownDisplay(result = it)
            }
        }
    }
}

fun startScanning(
    activity: ComponentActivity,
    scannerLauncher: androidx.activity.result.ActivityResultLauncher<IntentSenderRequest>
) {
    val options = GmsDocumentScannerOptions.Builder()
        .setScannerMode(GmsDocumentScannerOptions.SCANNER_MODE_FULL)
        .setGalleryImportAllowed(true)
        .setPageLimit(100)
        .setResultFormats(
            GmsDocumentScannerOptions.RESULT_FORMAT_JPEG,
            GmsDocumentScannerOptions.RESULT_FORMAT_PDF
        )
        .build()

    val scanner = GmsDocumentScanning.getClient(options)



    scanner.getStartScanIntent(activity).addOnSuccessListener { intentSender ->
        val request = IntentSenderRequest.Builder(intentSender).build()
        scannerLauncher.launch(request)
    }.addOnFailureListener { e ->
        Log.e("ScanError", "Error starting scanner", e)
    }
}

fun saveImagesAsPdf(context: Context, imageUris: List<Uri>) {
    val pdfDocument = PdfDocument()
    val resolver = context.contentResolver

    imageUris.forEachIndexed { index, uri ->
        val bitmap = MediaStore.Images.Media.getBitmap(resolver, uri)
        val pageInfo = PdfDocument.PageInfo.Builder(bitmap.width, bitmap.height, index + 1).create()
        val page = pdfDocument.startPage(pageInfo)
        page.canvas.drawBitmap(bitmap, 0f, 0f, null)
        pdfDocument.finishPage(page)
    }

    val fileName = "Scanned_Document_${System.currentTimeMillis()}.pdf"

    val contentValues = ContentValues().apply {
        put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
        put(MediaStore.MediaColumns.MIME_TYPE, "application/pdf")
        put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOCUMENTS)
    }


    val uri = resolver.insert(MediaStore.Files.getContentUri("external"), contentValues)

    uri?.let {
        try {
            resolver.openOutputStream(it)?.use { outputStream ->
                pdfDocument.writeTo(outputStream)
                Toast.makeText(context, "PDF saved in Documents!", Toast.LENGTH_SHORT).show()
            }
        } catch (e: IOException) {
            Toast.makeText(context, "Error saving PDF!", Toast.LENGTH_SHORT).show()
        } finally {
            pdfDocument.close()
        }
    }
}
fun Uri.getFileName(context: Context): String {
    val returnCursor = context.contentResolver.query(this, null, null, null, null)
    returnCursor?.use { cursor ->
        val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
        if (cursor.moveToFirst() && nameIndex != -1) {
            return cursor.getString(nameIndex)
        }
    }
    return "Unknown File"
}
@Composable
fun GeminiMarkdownDisplay(result: String) {
    MarkdownText(
        markdown = result,
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(16.dp)
    )

}
class EvaluationState : ViewModel() {
    var questionPaperUri by mutableStateOf<Uri?>(null)
    var answerSheetUri by mutableStateOf<Uri?>(null)
    var evaluationResult by mutableStateOf<String?>(null)
    var isLoading by mutableStateOf(false)
    var sliderValues = mutableStateMapOf<String, Float>()
}



class StudentState {
    private val _students = mutableStateListOf<Student>()
    val students: List<Student> get() = _students

    fun addStudent(student: Student) {
        _students.add(student)
    }

    fun removeStudent(rollNumber: String) {
        _students.removeIf { it.rollNumber == rollNumber }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AddStudentScreen(
    navController: NavController,
    studentState: StudentState
) {
    var rollNumber by remember { mutableStateOf("") }
    var name by remember { mutableStateOf("") }
    var email by remember { mutableStateOf("") }
    var searchQuery by remember { mutableStateOf("") }

    val filteredStudents = studentState.students.filter {
        it.name.contains(searchQuery, ignoreCase = true) ||
                it.rollNumber.contains(searchQuery, ignoreCase = true)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Add Student") },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary
                ),
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Back",
                            tint = Color.White
                        )
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .padding(paddingValues)
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .padding(16.dp)
        ) {
            // Search Bar
            TextField(
                value = searchQuery,
                onValueChange = { searchQuery = it },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp),
                placeholder = { Text("Search students...") },
                leadingIcon = {
                    Icon(
                        imageVector = Icons.Default.Search,
                        contentDescription = "Search"
                    )
                },
                shape = RoundedCornerShape(8.dp),
                colors = TextFieldDefaults.colors(
                    focusedContainerColor = MaterialTheme.colorScheme.surface,
                    unfocusedContainerColor = MaterialTheme.colorScheme.surface,
                    focusedIndicatorColor = Color.Transparent,
                    unfocusedIndicatorColor = Color.Transparent
                )
            )

            // Student Data Entry Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(12.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surface
                ),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        text = "Student Data",
                        style = MaterialTheme.typography.headlineSmall,
                        color = MaterialTheme.colorScheme.onSurface
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(
                        text = "Add New Student",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.8f)
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    OutlinedTextField(
                        value = rollNumber,
                        onValueChange = { rollNumber = it },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Roll Number") },
                        placeholder = { Text("Enter roll number") },
                        singleLine = true
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    OutlinedTextField(
                        value = name,
                        onValueChange = { name = it },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Name") },
                        placeholder = { Text("Enter full name") },
                        singleLine = true
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    OutlinedTextField(
                        value = email,
                        onValueChange = { email = it },
                        modifier = Modifier.fillMaxWidth(),
                        label = { Text("Email") },
                        placeholder = { Text("Enter email address") },
                        singleLine = true
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Button(
                            onClick = {
                                if (rollNumber.isNotBlank() && name.isNotBlank()) {
                                    studentState.addStudent(
                                        Student(rollNumber.trim(), name.trim(), email.trim())
                                    )
                                    rollNumber = ""
                                    name = ""
                                    email = ""
                                }
                            },
                            modifier = Modifier.weight(1f),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.primary
                            )
                        ) {
                            Text("Save")
                        }

                        OutlinedButton(
                            onClick = { navController.popBackStack() },
                            modifier = Modifier.weight(1f)
                        ) {
                            Text("Cancel")
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Student List
            if (filteredStudents.isNotEmpty()) {
                Text(
                    text = "Students List",
                    style = MaterialTheme.typography.titleLarge,
                    modifier = Modifier.padding(vertical = 8.dp)
                )

                LazyColumn(
                    modifier = Modifier.fillMaxWidth(),
                    contentPadding = PaddingValues(bottom = 100.dp)
                ) {
                    items(filteredStudents) { student ->
                        StudentListItem(
                            student = student,
                            onDelete = { studentState.removeStudent(student.rollNumber) }
                        )
                        Divider()
                    }
                }
            } else {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 32.dp),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Icon(
                        imageVector = Icons.Default.Person,
                        contentDescription = "No students",
                        tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                        modifier = Modifier.size(48.dp)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "No student data available. Add your first student to get started.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ParameterSliderScreen(navController: NavController, evalState: EvaluationState)
{
    val parameters = listOf(
        "Correctness",
        "Steps",
        "Accuracy",
        "Relevance",
        "Coherence",
        "Presentation"
    )

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "Adjust Evaluation Parameters",
                        style = MaterialTheme.typography.titleLarge,
                        modifier = Modifier.padding(bottom = 16.dp)
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFF5D4A79)
                )
            )
        },
        containerColor = Color(0xFFFDFDFF)
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .padding(paddingValues)
                .padding(16.dp)
        ) {
            parameters.forEach { parameter ->
                val value = evalState.sliderValues.getOrPut(parameter) { 50f }

                Text(text = "$parameter: ${value.toInt()}")
                Slider(
                    value = value,
                    onValueChange = { newValue ->
                        evalState.sliderValues[parameter] = newValue
                    },
                    valueRange = 0f..100f
                )
                Spacer(modifier = Modifier.height(16.dp))
            }

            Button(
                onClick = {
                    navController.popBackStack()
                },
                modifier = Modifier.align(Alignment.CenterHorizontally)
            ) {
                Text("Submit")
            }
        }
    }
}

@Composable
fun AssignmentTypeSelector(
    selectedType: String,
    onTypeChange: (String) -> Unit,
    customText: String
) {
    val options = listOf("Homework", "Test", "Assignment", "Custom")
    Column {
        Text("Assignment Type")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            options.forEach { type ->
                FilterChip(
                    selected = selectedType == type,
                    onClick = { onTypeChange(type) },
                    label = { Text(type) }
                )
            }
        }
        if (selectedType == "Custom") {
            OutlinedTextField(
                value = customText,
                onValueChange = onTypeChange,
                label = { Text("Custom Type") },
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}


@Composable
fun StudentListItem(
    student: Student,
    onDelete: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column {
            Text(
                text = student.name,
                style = MaterialTheme.typography.bodyLarge
            )
            Text(
                text = "Roll No: ${student.rollNumber}",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )
            if (student.email.isNotBlank()) {
                Text(
                    text = student.email,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                )
            }
        }

        IconButton(onClick = onDelete) {
            Icon(
                imageVector = Icons.Default.Delete,
                contentDescription = "Delete",
                tint = MaterialTheme.colorScheme.error
            )
        }
    }
}
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LessonPlanScreen(navController: NavController) {
    val context = LocalContext.current
    var useHours by remember { mutableStateOf(true) }
    var isLoading by remember { mutableStateOf(false) }
    var subject by remember { mutableStateOf("") }
    var topics by remember { mutableStateOf("") }
    var hours by remember { mutableStateOf("") }
    var toggleHours by remember { mutableStateOf(false) }
    var output by remember { mutableStateOf("") }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Lesson Planner") },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary
                ),
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Back",
                            tint = Color.White
                        )
                    }
                }
            )
        },
        containerColor = Color(0xFFFDFDFF)
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(24.dp, Alignment.Top),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(32.dp))

            OutlinedTextField(
                value = subject,
                onValueChange = { subject = it },
                label = { Text("Subject") },
                modifier = Modifier.fillMaxWidth()
            )

            OutlinedTextField(
                value = topics,
                onValueChange = { topics = it },
                label = { Text("Topics (comma-separated)") },
                modifier = Modifier.fillMaxWidth()
            )

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(5.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
            ) {
                Checkbox(
                    checked = toggleHours,
                    onCheckedChange = { toggleHours = it }
                )

                Text(
                    "Specify total time?",
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.widthIn(max = 140.dp)
                )

                Text("Use:")

                FilterChip(
                    selected = useHours,
                    onClick = { useHours = true },
                    label = { Text("Hours") },
                    modifier = Modifier.defaultMinSize(minWidth = 70.dp)
                )

                FilterChip(
                    selected = !useHours,
                    onClick = { useHours = false },
                    label = { Text("Classes") },
                    modifier = Modifier.defaultMinSize(minWidth = 70.dp)
                )
            }


            if (toggleHours) {
                OutlinedTextField(
                    value = hours,
                    onValueChange = { hours = it },
                    label = { Text(if (useHours) "Total Hours" else "Total Classes") },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.fillMaxWidth()
                )
            }


            Button(
                onClick = {
                    val prompt = buildLessonPrompt(subject, topics, hours.takeIf { toggleHours })
                    isLoading = true
                    generateLessonPlan(prompt) {
                        output = it
                        isLoading = false
                    }
                },
                modifier = Modifier.fillMaxWidth(),
                enabled = !isLoading
            ) {
                if (isLoading) {
                    CircularProgressIndicator(
                        color = Color.White,
                        modifier = Modifier.size(20.dp)
                    )
                } else {
                    Text("Generate Lesson Plann")
                }
            }


            if (output.isNotEmpty()) {
                Spacer(modifier = Modifier.height(16.dp))
                GeneratedAssignmentDialog(
                    titleText = "Generated Lesson Plan",
                    text = output,
                    onDismiss = { output = "" }
                )

            }
        }
    }
}

fun buildLessonPrompt(subject: String, topics: String, hours: String?, useHours: Boolean = true): String {
    val timeUnit = if (useHours) "hours" else "classes"
    return """
        You are a professional educator creating a structured lesson plan.

        Subject: $subject
        Topics: $topics
        ${hours?.let { "Total Duration: $it $timeUnit" } ?: ""}
        
        Create a well-organized plan with:
        - Weekly breakdown
        - Objectives per topic
        - Suggested teaching methods
        - Time distribution

        Use plain text, no markdown or emojis. Keep it clear and concise.
    """.trimIndent()
}


fun generateLessonPlan(prompt: String, callback: (String) -> Unit) {
    CoroutineScope(Dispatchers.IO).launch {
        try {
            val model = GenerativeModel(
                modelName = "gemini-1.5-flash",
                apiKey = xxxxxxxxxxx"
            )












            val response = model.generateContent(content { text(prompt) })
            withContext(Dispatchers.Main) {
                callback(response.text ?: "No output")
            }
        } catch (e: Exception) {
            Log.e("LessonPlanGen", "Error: ${e.localizedMessage}")
            withContext(Dispatchers.Main) {
                callback("Failed: ${e.localizedMessage}")
            }
        }
    }
}



