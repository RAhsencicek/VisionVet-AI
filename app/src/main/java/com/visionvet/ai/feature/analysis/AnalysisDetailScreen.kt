package com.visionvet.ai.feature.analysis

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.automirrored.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionvet.ai.core.database.model.Analysis
import com.visionvet.ai.core.database.model.ParasiteResult
import com.visionvet.ai.core.database.model.ParasiteType
import com.visionvet.ai.ui.theme.VisionVetAITheme
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AnalysisDetailScreen(
    analysis: Analysis,
    onDeleteAnalysis: () -> Unit,
    onUploadAnalysis: () -> Unit,
    onLearnMore: (ParasiteType) -> Unit,
    onNavigateBack: () -> Unit
) {
    var showDeleteDialog by remember { mutableStateOf(false) }
    var isUploading by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier.fillMaxSize()
    ) {
        // Top App Bar
        TopAppBar(
            title = { Text("Analysis Details") },
            navigationIcon = {
                IconButton(onClick = onNavigateBack) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                }
            }
        )

        ScrollView(
            analysis = analysis,
            isUploading = isUploading,
            onDeleteClick = { showDeleteDialog = true },
            onUploadClick = {
                isUploading = true
                onUploadAnalysis()
                isUploading = false
            },
            onLearnMoreClick = { onLearnMore(analysis.dominantParasite ?: ParasiteType.ASCARIS) }
        )
    }

    // Delete confirmation dialog
    if (showDeleteDialog) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            title = { Text("Delete Analysis") },
            text = { Text("Are you sure you want to delete this analysis? This action cannot be undone.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        showDeleteDialog = false
                        onDeleteAnalysis()
                        onNavigateBack()
                    }
                ) {
                    Text("Delete", color = MaterialTheme.colorScheme.error)
                }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = false }) {
                    Text("Cancel")
                }
            }
        )
    }
}

@Composable
private fun ScrollView(
    analysis: Analysis,
    isUploading: Boolean,
    onDeleteClick: () -> Unit,
    onUploadClick: () -> Unit,
    onLearnMoreClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Analysis Image
        analysis.imageBitmap?.let { bitmap ->
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            ) {
                Image(
                    bitmap = bitmap.asImageBitmap(),
                    contentDescription = "Analysis Image",
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp)
                        .clip(RoundedCornerShape(12.dp)),
                    contentScale = ContentScale.Crop
                )
            }
        }

        // Basic Information Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Analysis Information",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(bottom = 12.dp)
                )

                InfoRow("Location", analysis.location)
                InfoRow("Date", analysis.formattedDate)
                InfoRow("ID", analysis.id)
                InfoRow("Status", if (analysis.isUploaded) "Uploaded" else "Local")
            }
        }

        // Results Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Analysis Results",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(bottom = 12.dp)
                )

                if (analysis.results.isNotEmpty()) {
                    analysis.results.forEach { result ->
                        ParasiteResultItem(
                            result = result,
                            onLearnMore = onLearnMoreClick
                        )
                    }
                } else {
                    Text(
                        text = "No parasites detected",
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        // Notes Card
        if (analysis.notes.isNotEmpty()) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(
                        text = "Notes",
                        fontSize = 18.sp,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.padding(bottom = 8.dp)
                    )
                    Text(
                        text = analysis.notes,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        // Action Buttons
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = "Actions",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(bottom = 4.dp)
                )

                if (!analysis.isUploaded) {
                    Button(
                        onClick = onUploadClick,
                        modifier = Modifier.fillMaxWidth(),
                        enabled = !isUploading
                    ) {
                        if (isUploading) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(16.dp),
                                color = Color.White
                            )
                        } else {
                            Icon(
                                Icons.Default.CloudUpload,
                                contentDescription = null,
                                modifier = Modifier.padding(end = 8.dp)
                            )
                            Text("Upload to Cloud")
                        }
                    }
                }

                OutlinedButton(
                    onClick = onDeleteClick,
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = MaterialTheme.colorScheme.error
                    )
                ) {
                    Icon(
                        Icons.Default.Delete,
                        contentDescription = null,
                        modifier = Modifier.padding(end = 8.dp)
                    )
                    Text("Delete Analysis")
                }
            }
        }
    }
}

@Composable
private fun InfoRow(
    label: String,
    value: String
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f)
        )
        Text(
            text = value,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.weight(2f)
        )
    }
}

@Composable
private fun ParasiteResultItem(
    result: ParasiteResult,
    onLearnMore: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
        )
    ) {
        Column(
            modifier = Modifier.padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = result.type.displayName,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Medium
                )
                Text(
                    text = "${(result.confidence * 100).toInt()}%",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = getConfidenceColor(result.confidence)
                )
            }

            // Confidence bar
            LinearProgressIndicator(
                progress = result.confidence.toFloat(),
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp),
                color = getConfidenceColor(result.confidence)
            )

            TextButton(
                onClick = onLearnMore,
                modifier = Modifier.align(Alignment.End)
            ) {
                Text("Learn More")
            }
        }
    }
}

@Composable
private fun getConfidenceColor(confidence: Double): Color {
    return when {
        confidence >= 0.8 -> Color(0xFF34C759) // Green
        confidence >= 0.6 -> Color(0xFFFF9500) // Orange
        else -> Color(0xFFFF3B30) // Red
    }
}

// MARK: - Preview Functions (Swift benzeri)
@Preview(showBackground = true)
@Composable
fun AnalysisDetailScreenPreview() {
    VisionVetAITheme {
        AnalysisDetailScreen(
            analysis = getSampleAnalysis(),
            onDeleteAnalysis = {},
            onUploadAnalysis = {},
            onLearnMore = {},
            onNavigateBack = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun AnalysisDetailScreenDarkPreview() {
    VisionVetAITheme(darkTheme = true) {
        AnalysisDetailScreen(
            analysis = getSampleAnalysis(),
            onDeleteAnalysis = {},
            onUploadAnalysis = {},
            onLearnMore = {},
            onNavigateBack = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun ParasiteResultItemPreview() {
    VisionVetAITheme {
        ParasiteResultItem(
            result = ParasiteResult(ParasiteType.ASCARIS, 0.85, System.currentTimeMillis()),
            onLearnMore = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun InfoRowPreview() {
    VisionVetAITheme {
        Column(modifier = Modifier.padding(16.dp)) {
            InfoRow("Location", "Lab Room 1")
            InfoRow("Date", "Jan 15, 2024 14:30")
            InfoRow("Status", "Uploaded")
        }
    }
}

// Sample data for previews
private fun getSampleAnalysis(): Analysis {
    val currentTime = System.currentTimeMillis()
    return Analysis(
        id = "sample-123",
        userId = "user1",
        location = "Veterinary Lab - Room A",
        timestamp = currentTime,
        notes = "Routine parasite screening for dairy cattle. Sample collected from healthy appearing animal during regular health check.",
        results = listOf(
            ParasiteResult(ParasiteType.ASCARIS, 0.85, currentTime),
            ParasiteResult(ParasiteType.HOOKWORM, 0.32, currentTime)
        ),
        isUploaded = false
    )
}
