package com.visionvet.ai.feature.bacterial

import android.net.Uri
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import coil.compose.rememberAsyncImagePainter
import com.visionvet.ai.core.database.VisionVetDatabase
import com.visionvet.ai.core.database.model.BacterialResult
import com.visionvet.ai.core.database.model.BacterialPrediction
import com.visionvet.ai.core.database.repository.BacterialRepository
import com.visionvet.ai.ui.components.GlassmorphicCard
import com.visionvet.ai.ui.components.GradientButton
import com.visionvet.ai.ui.theme.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BacterialResultScreen(
    resultId: String,
    onNavigateBack: () -> Unit = {},
    onNavigateToHistory: () -> Unit = {}
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    
    var result by remember { mutableStateOf<BacterialResult?>(null) }
    var notes by remember { mutableStateOf("") }
    var location by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(true) }
    var isSaving by remember { mutableStateOf(false) }
    var showSaveConfirmation by remember { mutableStateOf(false) }
    
    val database = remember { VisionVetDatabase.getDatabase(context) }
    val repository = remember { BacterialRepository(database.bacterialResultDao()) }
    
    // Load result
    LaunchedEffect(resultId) {
        try {
            result = repository.getResultById(resultId)
            result?.let {
                notes = it.notes
                location = it.location
            }
        } catch (e: Exception) {
            android.util.Log.e("BacterialResult", "Failed to load result", e)
        } finally {
            isLoading = false
        }
    }
    
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(DarkBackground)
    ) {
        if (isLoading) {
            LoadingState()
        } else if (result == null) {
            ErrorState(onNavigateBack = onNavigateBack)
        } else {
            Column(modifier = Modifier.fillMaxSize()) {
                // Top Bar with Image Preview
                ResultHeader(
                    result = result!!,
                    onNavigateBack = onNavigateBack,
                    onNavigateToHistory = onNavigateToHistory
                )
                
                // Content
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(rememberScrollState())
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    // Primary Result Card
                    PrimaryResultCard(result = result!!)
                    
                    // All Predictions
                    PredictionsCard(predictions = result?.predictions ?: emptyList())
                    
                    // Analysis Info
                    AnalysisInfoCard(result = result!!)
                    
                    // Notes Section
                    NotesCard(
                        notes = notes,
                        onNotesChange = { notes = it }
                    )
                    
                    // Location Section
                    LocationCard(
                        location = location,
                        onLocationChange = { location = it }
                    )
                    
                    // Save Button
                    GradientButton(
                        text = if (isSaving) "Saving..." else "Save Changes",
                        onClick = {
                            scope.launch {
                                try {
                                    isSaving = true
                                    result?.let { currentResult ->
                                        val updated = currentResult.copy(
                                            notes = notes,
                                            location = location
                                        )
                                        repository.updateResult(updated)
                                        showSaveConfirmation = true
                                    }
                                } catch (e: Exception) {
                                    android.util.Log.e("BacterialResult", "Failed to save", e)
                                } finally {
                                    isSaving = false
                                }
                            }
                        },
                        enabled = !isSaving,
                        modifier = Modifier.fillMaxWidth(),
                        leadingContent = {
                            if (isSaving) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(20.dp),
                                    color = Color.White,
                                    strokeWidth = 2.dp
                                )
                            } else {
                                Icon(
                                    Icons.Default.Save, 
                                    contentDescription = null,
                                    tint = Color.White
                                )
                            }
                        }
                    )
                    
                    // Save Confirmation
                    AnimatedVisibility(
                        visible = showSaveConfirmation,
                        enter = fadeIn() + slideInVertically(),
                        exit = fadeOut() + slideOutVertically()
                    ) {
                        LaunchedEffect(showSaveConfirmation) {
                            if (showSaveConfirmation) {
                                delay(3000)
                                showSaveConfirmation = false
                            }
                        }
                        
                        SaveConfirmationCard()
                    }
                    
                    Spacer(modifier = Modifier.height(100.dp))
                }
            }
        }
    }
}

@Composable
private fun LoadingState() {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            CircularProgressIndicator(
                color = BacteriaBlue,
                strokeWidth = 3.dp,
                modifier = Modifier.size(48.dp)
            )
            Text(
                "Loading result...",
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
private fun ErrorState(onNavigateBack: () -> Unit) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(24.dp),
            modifier = Modifier.padding(32.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .background(
                        color = Color.Red.copy(alpha = 0.2f),
                        shape = CircleShape
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.Error,
                    contentDescription = null,
                    modifier = Modifier.size(40.dp),
                    tint = Color.Red
                )
            }
            
            Text(
                "Result Not Found",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            
            Text(
                "The analysis result you're looking for could not be found.",
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White.copy(alpha = 0.6f),
                textAlign = TextAlign.Center
            )
            
            GradientButton(
                text = "Go Back",
                onClick = onNavigateBack,
                modifier = Modifier.width(200.dp),
                leadingContent = {
                    Icon(
                        Icons.AutoMirrored.Filled.ArrowBack, 
                        contentDescription = null,
                        tint = Color.White
                    )
                }
            )
        }
    }
}

@Composable
private fun ResultHeader(
    result: BacterialResult,
    onNavigateBack: () -> Unit,
    onNavigateToHistory: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(280.dp)
    ) {
        // Background Image or Gradient
        if (result.imagePath.isNotEmpty()) {
            Image(
                painter = rememberAsyncImagePainter(Uri.parse(result.imagePath)),
                contentDescription = null,
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop
            )
            
            // Gradient overlay
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        brush = Brush.verticalGradient(
                            colors = listOf(
                                Color.Black.copy(alpha = 0.4f),
                                Color.Black.copy(alpha = 0.2f),
                                DarkBackground
                            )
                        )
                    )
            )
        } else {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        brush = Brush.verticalGradient(
                            colors = listOf(
                                DeepBlue.copy(alpha = 0.4f),
                                DarkBackground
                            )
                        )
                    )
            )
        }
        
        // Top Navigation
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 48.dp, start = 16.dp, end = 16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            IconButton(
                onClick = onNavigateBack,
                modifier = Modifier
                    .size(44.dp)
                    .background(
                        color = Color.Black.copy(alpha = 0.3f),
                        shape = CircleShape
                    )
            ) {
                Icon(
                    Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "Back",
                    tint = Color.White
                )
            }
            
            IconButton(
                onClick = onNavigateToHistory,
                modifier = Modifier
                    .size(44.dp)
                    .background(
                        color = Color.Black.copy(alpha = 0.3f),
                        shape = CircleShape
                    )
            ) {
                Icon(
                    Icons.Default.History,
                    contentDescription = "History",
                    tint = Color.White
                )
            }
        }
        
        // Result Badge at bottom
        result.topPrediction?.let { top ->
            Box(
                modifier = Modifier
                    .align(Alignment.BottomStart)
                    .padding(16.dp)
            ) {
                Row(
                    modifier = Modifier
                        .background(
                            color = Color.Black.copy(alpha = 0.5f),
                            shape = RoundedCornerShape(12.dp)
                        )
                        .padding(horizontal = 16.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Icon(
                        Icons.Default.Science,
                        contentDescription = null,
                        tint = BacteriaBlue,
                        modifier = Modifier.size(20.dp)
                    )
                    Text(
                        top.displayName,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                }
            }
        }
    }
}

@Composable
private fun PrimaryResultCard(result: BacterialResult) {
    result.topPrediction?.let { top ->
        GlassmorphicCard(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            "Primary Detection",
                            style = MaterialTheme.typography.labelMedium,
                            color = Color.White.copy(alpha = 0.6f)
                        )
                        Spacer(Modifier.height(4.dp))
                        Text(
                            top.displayName,
                            style = MaterialTheme.typography.headlineSmall,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                    }
                    
                    // Confidence Badge
                    Box(
                        modifier = Modifier
                            .background(
                                brush = if (result.isHighConfidence)
                                    Brush.linearGradient(listOf(MicrobeGreen, MicrobeGreen.copy(alpha = 0.7f)))
                                else
                                    Brush.linearGradient(listOf(Color.Yellow, Color.Yellow.copy(alpha = 0.7f))),
                                shape = RoundedCornerShape(12.dp)
                            )
                            .padding(horizontal = 16.dp, vertical = 8.dp)
                    ) {
                        Text(
                            top.formattedConfidence,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = Color.Black
                        )
                    }
                }
                
                // Warning for low confidence
                if (!result.isHighConfidence) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(
                                color = Color.Yellow.copy(alpha = 0.1f),
                                shape = RoundedCornerShape(8.dp)
                            )
                            .padding(12.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            Icons.Default.Warning,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp),
                            tint = Color.Yellow
                        )
                        Text(
                            "Low confidence - consider retaking image",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color.Yellow
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun PredictionsCard(predictions: List<BacterialPrediction>) {
    GlassmorphicCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Icon(
                    Icons.Outlined.Analytics,
                    contentDescription = null,
                    tint = BacteriaBlue,
                    modifier = Modifier.size(24.dp)
                )
                Text(
                    "All Predictions",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
            }
            
            predictions.take(5).forEachIndexed { index, prediction ->
                PredictionItem(
                    prediction = prediction,
                    rank = index + 1
                )
            }
        }
    }
}

@Composable
private fun PredictionItem(
    prediction: BacterialPrediction,
    rank: Int
) {
    val color = when (rank) {
        1 -> BacteriaBlue
        2 -> ElectricPurple
        3 -> MicrobeGreen
        else -> Color.White.copy(alpha = 0.5f)
    }
    
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Box(
                    modifier = Modifier
                        .size(24.dp)
                        .background(
                            color = color.copy(alpha = 0.2f),
                            shape = CircleShape
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        "#$rank",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = color
                    )
                }
                
                Text(
                    prediction.displayName,
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.White
                )
            }
            
            Text(
                prediction.formattedConfidence,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.SemiBold,
                color = color
            )
        }
        
        // Progress bar
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(4.dp)
                .clip(RoundedCornerShape(2.dp))
                .background(Color.White.copy(alpha = 0.1f))
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(prediction.confidence.toFloat())
                    .fillMaxHeight()
                    .background(
                        brush = Brush.horizontalGradient(
                            colors = listOf(color, color.copy(alpha = 0.5f))
                        )
                    )
            )
        }
    }
}

@Composable
private fun AnalysisInfoCard(result: BacterialResult) {
    GlassmorphicCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Icon(
                    Icons.Outlined.Info,
                    contentDescription = null,
                    tint = BacteriaBlue,
                    modifier = Modifier.size(24.dp)
                )
                Text(
                    "Analysis Information",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
            }
            
            InfoItem(
                icon = Icons.Outlined.CalendarToday,
                label = "Date",
                value = result.formattedDate
            )
            
            InfoItem(
                icon = Icons.Outlined.Fingerprint,
                label = "Analysis ID",
                value = result.id.take(8).uppercase()
            )
            
            InfoItem(
                icon = Icons.Outlined.Science,
                label = "Predictions",
                value = "${result.predictions.size} species identified"
            )
        }
    }
}

@Composable
private fun InfoItem(
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    label: String,
    value: String
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Icon(
            icon,
            contentDescription = null,
            modifier = Modifier.size(20.dp),
            tint = Color.White.copy(alpha = 0.5f)
        )
        
        Column {
            Text(
                label,
                style = MaterialTheme.typography.bodySmall,
                color = Color.White.copy(alpha = 0.5f)
            )
            Text(
                value,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                color = Color.White
            )
        }
    }
}

@Composable
private fun NotesCard(
    notes: String,
    onNotesChange: (String) -> Unit
) {
    GlassmorphicCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Icon(
                    Icons.Outlined.Edit,
                    contentDescription = null,
                    tint = ElectricPurple,
                    modifier = Modifier.size(24.dp)
                )
                Text(
                    "Notes",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
            }
            
            OutlinedTextField(
                value = notes,
                onValueChange = onNotesChange,
                modifier = Modifier.fillMaxWidth(),
                placeholder = {
                    Text(
                        "Add notes about this sample...",
                        color = Color.White.copy(alpha = 0.4f)
                    )
                },
                minLines = 3,
                maxLines = 5,
                colors = OutlinedTextFieldDefaults.colors(
                    unfocusedTextColor = Color.White,
                    focusedTextColor = Color.White,
                    unfocusedBorderColor = Color.White.copy(alpha = 0.2f),
                    focusedBorderColor = BacteriaBlue,
                    cursorColor = BacteriaBlue
                ),
                shape = RoundedCornerShape(12.dp)
            )
        }
    }
}

@Composable
private fun LocationCard(
    location: String,
    onLocationChange: (String) -> Unit
) {
    GlassmorphicCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Icon(
                    Icons.Outlined.LocationOn,
                    contentDescription = null,
                    tint = MicrobeGreen,
                    modifier = Modifier.size(24.dp)
                )
                Text(
                    "Location",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
            }
            
            OutlinedTextField(
                value = location,
                onValueChange = onLocationChange,
                modifier = Modifier.fillMaxWidth(),
                placeholder = {
                    Text(
                        "Sample location...",
                        color = Color.White.copy(alpha = 0.4f)
                    )
                },
                leadingIcon = {
                    Icon(
                        Icons.Outlined.MyLocation,
                        contentDescription = null,
                        tint = Color.White.copy(alpha = 0.5f)
                    )
                },
                singleLine = true,
                colors = OutlinedTextFieldDefaults.colors(
                    unfocusedTextColor = Color.White,
                    focusedTextColor = Color.White,
                    unfocusedBorderColor = Color.White.copy(alpha = 0.2f),
                    focusedBorderColor = BacteriaBlue,
                    cursorColor = BacteriaBlue
                ),
                shape = RoundedCornerShape(12.dp)
            )
        }
    }
}

@Composable
private fun SaveConfirmationCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MicrobeGreen.copy(alpha = 0.15f)
        ),
        shape = RoundedCornerShape(16.dp)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(40.dp)
                    .background(
                        color = MicrobeGreen.copy(alpha = 0.2f),
                        shape = CircleShape
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    Icons.Default.CheckCircle,
                    contentDescription = null,
                    tint = MicrobeGreen,
                    modifier = Modifier.size(24.dp)
                )
            }
            
            Text(
                "Changes saved successfully!",
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                color = MicrobeGreen
            )
        }
    }
}
