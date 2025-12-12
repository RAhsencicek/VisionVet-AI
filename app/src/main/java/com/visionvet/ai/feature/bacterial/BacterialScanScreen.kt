package com.visionvet.ai.feature.bacterial

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.visionvet.ai.ml.bacterial.BacterialClassifier
import com.visionvet.ai.core.database.VisionVetDatabase
import com.visionvet.ai.core.database.repository.BacterialRepository
import com.visionvet.ai.core.database.model.BacterialResult
import com.visionvet.ai.core.database.model.BacterialPrediction
import kotlinx.coroutines.launch
import java.io.File
import java.util.UUID

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BacterialScanScreen(
    onNavigateToResult: (String) -> Unit = {},
    onNavigateBack: () -> Unit = {}
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    var capturedImage by remember { mutableStateOf<Bitmap?>(null) }
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var isProcessing by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }
    
    val classifier = remember { BacterialClassifier(context) }
    val database = remember { VisionVetDatabase.getDatabase(context) }
    val repository = remember { BacterialRepository(database.bacterialResultDao()) }
    
    // Initialize classifier
    LaunchedEffect(Unit) {
        try {
            classifier.initialize()
        } catch (e: Exception) {
            errorMessage = "Failed to initialize: ${e.message}"
        }
    }
    
    // Camera permission launcher
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
        if (!isGranted) {
            errorMessage = "Camera permission is required"
        }
    }
    
    // Camera launcher
    val cameraLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success && imageUri != null) {
            try {
                val bitmap = BitmapFactory.decodeStream(
                    context.contentResolver.openInputStream(imageUri!!)
                )
                capturedImage = bitmap
                errorMessage = null
            } catch (e: Exception) {
                errorMessage = "Failed to load image: ${e.message}"
            }
        }
    }
    
    // Gallery launcher
    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            try {
                val bitmap = BitmapFactory.decodeStream(
                    context.contentResolver.openInputStream(it)
                )
                capturedImage = bitmap
                imageUri = it
                errorMessage = null
            } catch (e: Exception) {
                errorMessage = "Failed to load image: ${e.message}"
            }
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Bacterial Colony Scan") },
                navigationIcon = {
                    IconButton(onClick = onNavigateBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back")
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Instructions Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(
                        "Instructions",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(Modifier.height(8.dp))
                    Text("1. Take a photo or select from gallery", style = MaterialTheme.typography.bodyMedium)
                    Text("2. Ensure good lighting and focus", style = MaterialTheme.typography.bodyMedium)
                    Text("3. Center the bacterial colony", style = MaterialTheme.typography.bodyMedium)
                }
            }
            
            // Image Capture Buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = {
                        if (hasCameraPermission) {
                            val photoFile = File(
                                context.cacheDir,
                                "bacterial_${System.currentTimeMillis()}.jpg"
                            )
                            imageUri = FileProvider.getUriForFile(
                                context,
                                "${context.packageName}.fileprovider",
                                photoFile
                            )
                            cameraLauncher.launch(imageUri!!)
                        } else {
                            permissionLauncher.launch(Manifest.permission.CAMERA)
                        }
                    },
                    modifier = Modifier.weight(1f),
                    enabled = !isProcessing
                ) {
                    Icon(Icons.Default.CameraAlt, null)
                    Spacer(Modifier.width(8.dp))
                    Text("Camera")
                }
                
                Button(
                    onClick = { galleryLauncher.launch("image/*") },
                    modifier = Modifier.weight(1f),
                    enabled = !isProcessing
                ) {
                    Icon(Icons.Default.PhotoLibrary, null)
                    Spacer(Modifier.width(8.dp))
                    Text("Gallery")
                }
            }
            
            // Captured Image
            capturedImage?.let { bitmap ->
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text("Captured Image", fontWeight = FontWeight.Bold)
                        Spacer(Modifier.height(8.dp))
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = "Captured image",
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(300.dp)
                        )
                    }
                }
                
                // Analyze Button
                Button(
                    onClick = {
                        scope.launch {
                            try {
                                isProcessing = true
                                errorMessage = null
                                
                                // Run classification
                                val predictions = classifier.classifyTopK(bitmap, k = 5)
                                
                                // Save to database
                                val bacterialPredictions = predictions.map { pred ->
                                    BacterialPrediction(
                                        className = pred.className,
                                        displayName = pred.className.replace('_', ' '),
                                        confidence = pred.confidence,
                                        probability = pred.probability
                                    )
                                }
                                
                                val result = BacterialResult(
                                    id = UUID.randomUUID().toString(),
                                    userId = "current_user",
                                    timestamp = System.currentTimeMillis(),
                                    imagePath = imageUri?.toString() ?: "",
                                    predictions = bacterialPredictions,
                                    notes = "",
                                    location = ""
                                )
                                
                                repository.insertResult(result)
                                
                                // Navigate to result screen
                                onNavigateToResult(result.id)
                                
                            } catch (e: Exception) {
                                errorMessage = "Analysis failed: ${e.message}"
                            } finally {
                                isProcessing = false
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    enabled = !isProcessing
                ) {
                    if (isProcessing) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(24.dp),
                            color = MaterialTheme.colorScheme.onPrimary
                        )
                        Spacer(Modifier.width(8.dp))
                        Text("Analyzing...")
                    } else {
                        Icon(Icons.Default.Science, null)
                        Spacer(Modifier.width(8.dp))
                        Text("Analyze Colony")
                    }
                }
            }
            
            // Error Message
            errorMessage?.let { error ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Text(
                        error,
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.onErrorContainer
                    )
                }
            }
        }
    }
    
    DisposableEffect(Unit) {
        onDispose { classifier.close() }
    }
}
