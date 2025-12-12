package com.visionvet.ai.feature.scanner

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.visionvet.ai.ui.theme.VisionVetAITheme

enum class AnalysisType {
    PARASITE,
    MNIST
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NewScanScreen(
    onNavigateToCamera: (AnalysisType) -> Unit,
    onNavigateToDrawing: () -> Unit
) {
    var selectedAnalysisType by remember { mutableStateOf(AnalysisType.PARASITE) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Analysis type selection - Swift'teki Picker gibi
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Analysis Type",
                    fontSize = 16.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(bottom = 12.dp)
                )

                // Segmented control like picker - Swift'teki .segmented style
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(
                            MaterialTheme.colorScheme.surfaceVariant,
                            RoundedCornerShape(12.dp)
                        )
                        .padding(4.dp)
                ) {
                    AnalysisTypeButton(
                        text = "Parasite Analysis",
                        isSelected = selectedAnalysisType == AnalysisType.PARASITE,
                        onClick = { selectedAnalysisType = AnalysisType.PARASITE },
                        modifier = Modifier.weight(1f)
                    )

                    AnalysisTypeButton(
                        text = "Digit Recognition",
                        isSelected = selectedAnalysisType == AnalysisType.MNIST,
                        onClick = { selectedAnalysisType = AnalysisType.MNIST },
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }

        // Content based on selected analysis type
        when (selectedAnalysisType) {
            AnalysisType.PARASITE -> ParasiteScanView(onNavigateToCamera = onNavigateToCamera)
            AnalysisType.MNIST -> MNISTScanView(
                onNavigateToCamera = onNavigateToCamera,
                onNavigateToDrawing = onNavigateToDrawing
            )
        }
    }
}

@Composable
private fun AnalysisTypeButton(
    text: String,
    isSelected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    Button(
        onClick = onClick,
        modifier = modifier.padding(2.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = if (isSelected)
                MaterialTheme.colorScheme.primary
            else
                Color.Transparent,
            contentColor = if (isSelected)
                Color.White
            else
                MaterialTheme.colorScheme.onSurfaceVariant
        ),
        shape = RoundedCornerShape(8.dp),
        elevation = ButtonDefaults.buttonElevation(
            defaultElevation = if (isSelected) 2.dp else 0.dp
        )
    ) {
        Text(
            text = text,
            fontSize = 12.sp,
            fontWeight = if (isSelected) FontWeight.SemiBold else FontWeight.Normal
        )
    }
}

@Composable
private fun ParasiteScanView(
    onNavigateToCamera: (AnalysisType) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        Text(
            text = "Parasite Analysis",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(vertical = 16.dp)
        )

        // Description
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Instructions",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                Text(
                    text = "1. Prepare your microscopic sample\n" +
                            "2. Take a clear photo using the camera\n" +
                            "3. Wait for AI analysis results\n" +
                            "4. Review detected parasites",
                    fontSize = 14.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        // Camera button
        Button(
            onClick = { onNavigateToCamera(AnalysisType.PARASITE) },
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(12.dp)
        ) {
            Icon(
                Icons.Default.CameraAlt,
                contentDescription = null,
                modifier = Modifier.padding(end = 8.dp)
            )
            Text(
                text = "Start Camera Analysis",
                fontSize = 16.sp,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

@Composable
private fun MNISTScanView(
    onNavigateToCamera: (AnalysisType) -> Unit,
    onNavigateToDrawing: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        Text(
            text = "Digit Recognition",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(vertical = 16.dp)
        )

        // Description
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Choose Recognition Method",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                Text(
                    text = "Take a photo of handwritten digits or draw them directly on screen for AI recognition.",
                    fontSize = 14.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }

        // Buttons
        Column(
            modifier = Modifier.fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Button(
                onClick = { onNavigateToCamera(AnalysisType.MNIST) },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(
                    Icons.Default.CameraAlt,
                    contentDescription = null,
                    modifier = Modifier.padding(end = 8.dp)
                )
                Text(
                    text = "Camera Recognition",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold
                )
            }

            OutlinedButton(
                onClick = onNavigateToDrawing,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(
                    Icons.Default.Draw,
                    contentDescription = null,
                    modifier = Modifier.padding(end = 8.dp)
                )
                Text(
                    text = "Draw on Screen",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }
    }
}

// MARK: - Preview Functions (Swift benzeri)
@Preview(showBackground = true)
@Composable
fun NewScanScreenPreview() {
    VisionVetAITheme {
        NewScanScreen(
            onNavigateToCamera = {},
            onNavigateToDrawing = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun NewScanScreenDarkPreview() {
    VisionVetAITheme(darkTheme = true) {
        NewScanScreen(
            onNavigateToCamera = {},
            onNavigateToDrawing = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun ParasiteScanViewPreview() {
    VisionVetAITheme {
        ParasiteScanView(
            onNavigateToCamera = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun MNISTScanViewPreview() {
    VisionVetAITheme {
        MNISTScanView(
            onNavigateToCamera = {},
            onNavigateToDrawing = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun AnalysisTypeButtonPreview() {
    VisionVetAITheme {
        Row(
            modifier = Modifier.padding(16.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            AnalysisTypeButton(
                text = "Selected",
                isSelected = true,
                onClick = {},
                modifier = Modifier.weight(1f)
            )
            AnalysisTypeButton(
                text = "Unselected",
                isSelected = false,
                onClick = {},
                modifier = Modifier.weight(1f)
            )
        }
    }
}
