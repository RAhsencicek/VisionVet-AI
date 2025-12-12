package com.visionvet.ai.feature.bacterial

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Science
import androidx.compose.material.icons.filled.Save
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.visionvet.ai.ml.bacterial.BacterialClassifier
import com.visionvet.ai.core.database.VisionVetDatabase
import com.visionvet.ai.core.database.repository.BacterialRepository
import com.visionvet.ai.core.database.model.BacterialResult
import com.visionvet.ai.core.database.model.BacterialPrediction
import kotlinx.coroutines.launch
import java.util.UUID

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BacterialTestScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    var isInitialized by remember { mutableStateOf(false) }
    var isLoading by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var predictions by remember { mutableStateOf<List<BacterialClassifier.Prediction>>(emptyList()) }
    var testBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var savedResultId by remember { mutableStateOf<String?>(null) }
    
    val classifier = remember { BacterialClassifier(context) }
    val database = remember { VisionVetDatabase.getDatabase(context) }
    val repository = remember { BacterialRepository(database.bacterialResultDao()) }
    
    LaunchedEffect(Unit) {
        try {
            android.util.Log.d("BacterialTest", "Starting initialization...")
            isLoading = true
            classifier.initialize()
            isInitialized = true
            errorMessage = null
            android.util.Log.d("BacterialTest", "✓ Initialization successful")
        } catch (e: Exception) {
            errorMessage = "Initialization failed: ${e.message}"
            android.util.Log.e("BacterialTest", "✗ Initialization failed", e)
        } finally {
            isLoading = false
        }
    }
    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Bacterial Classifier Test") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = if (isInitialized) 
                        MaterialTheme.colorScheme.primaryContainer 
                    else 
                        MaterialTheme.colorScheme.errorContainer
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = "Model Status",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    
                    if (isLoading) {
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            CircularProgressIndicator(modifier = Modifier.size(24.dp))
                            Text("Initializing...")
                        }
                    } else if (isInitialized) {
                        Text("✓ Model loaded")
                        Text("✓ Ready for inference")
                    } else {
                        Text("✗ Not initialized")
                    }
                }
            }
            
            Button(
                onClick = {
                    scope.launch {
                        try {
                            isLoading = true
                            errorMessage = null
                            savedResultId = null
                            
                            val bitmap = createTestBitmap()
                            testBitmap = bitmap
                            
                            val results = classifier.classifyTopK(bitmap, k = 10)
                            predictions = results
                            android.util.Log.d("BacterialTest", "✓ Complete: ${results.size}")
                            
                        } catch (e: Exception) {
                            errorMessage = "Failed: ${e.message}"
                            android.util.Log.e("BacterialTest", "✗ Failed", e)
                        } finally {
                            isLoading = false
                        }
                    }
                },
                enabled = isInitialized && !isLoading,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(Icons.Default.Science, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text("Run Test")
            }
            
            if (predictions.isNotEmpty() && savedResultId == null) {
                Button(
                    onClick = {
                        scope.launch {
                            try {
                                val bacterialPreds = predictions.take(5).map { pred ->
                                    BacterialPrediction(
                                        className = pred.className,
                                        displayName = pred.className.replace('_', ' '),
                                        confidence = pred.confidence,
                                        probability = pred.probability
                                    )
                                }
                                
                                val result = BacterialResult(
                                    id = UUID.randomUUID().toString(),
                                    userId = "test_user",
                                    timestamp = System.currentTimeMillis(),
                                    imagePath = "/test/gradient.png",
                                    predictions = bacterialPreds,
                                    notes = "Test from BacterialTestScreen",
                                    location = "Test Lab"
                                )
                                
                                repository.insertResult(result)
                                savedResultId = result.id
                                
                                android.util.Log.d("BacterialTest", "✓ Saved: ${result.id}")
                                val count = repository.getResultCount("test_user")
                                android.util.Log.d("BacterialTest", "✓ Total in DB: $count")
                                
                            } catch (e: Exception) {
                                errorMessage = "Save failed: ${e.message}"
                                android.util.Log.e("BacterialTest", "✗ Save failed", e)
                            }
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.secondary
                    )
                ) {
                    Icon(Icons.Default.Save, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Save to Database")
                }
            }
            
            if (savedResultId != null) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.secondaryContainer
                    )
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("✓ Saved to Database", fontWeight = FontWeight.Bold)
                        Text("ID: $savedResultId", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
            
            testBitmap?.let { bitmap ->
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Test Image", style = MaterialTheme.typography.titleSmall)
                        Spacer(Modifier.height(8.dp))
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = null,
                            modifier = Modifier.size(224.dp).align(Alignment.CenterHorizontally)
                        )
                    }
                }
            }
            
            if (predictions.isNotEmpty()) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Top 10 Predictions", fontWeight = FontWeight.Bold)
                        Spacer(Modifier.height(8.dp))
                        
                        LazyColumn(
                            verticalArrangement = Arrangement.spacedBy(4.dp),
                            modifier = Modifier.heightIn(max = 400.dp)
                        ) {
                            items(predictions) { pred ->
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        pred.className.replace('_', ' '),
                                        modifier = Modifier.weight(1f)
                                    )
                                    Text(
                                        "${pred.confidence.toInt()}%",
                                        fontWeight = FontWeight.Bold,
                                        color = MaterialTheme.colorScheme.primary
                                    )
                                }
                            }
                        }
                    }
                }
            }
            
            errorMessage?.let { error ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Text(error, modifier = Modifier.padding(16.dp))
                }
            }
        }
    }
    
    DisposableEffect(Unit) {
        onDispose { classifier.close() }
    }
}

private fun createTestBitmap(): Bitmap {
    val size = 224
    val bitmap = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
    val pixels = IntArray(size * size)
    for (y in 0 until size) {
        for (x in 0 until size) {
            val r = (x * 255 / size)
            val g = (y * 255 / size)
            val b = ((x + y) * 255 / (size * 2))
            pixels[y * size + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
    }
    bitmap.setPixels(pixels, 0, size, 0, 0, size, size)
    return bitmap
}
