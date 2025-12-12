package com.visionvet.ai.feature.dashboard

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.automirrored.filled.ShowChart
import androidx.compose.material.icons.automirrored.filled.TrendingUp
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.visionvet.ai.core.database.model.Analysis
import com.visionvet.ai.core.database.model.AnalysisType
import com.visionvet.ai.core.database.model.ParasiteType
import com.visionvet.ai.ui.theme.VisionVetAITheme
import java.text.SimpleDateFormat
import java.util.*

data class StatCardData(
    val title: String,
    val value: String,
    val icon: ImageVector,
    val color: Color
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DashboardScreen(
    onNavigateToCamera: () -> Unit = {},
    onNavigateToHistory: () -> Unit = {},
    onNavigateToProfile: () -> Unit = {},
    onAnalysisClick: (String) -> Unit = {},
    onNavigateToMnist: () -> Unit = {},
    onNavigateToBacterialTest: () -> Unit = {},
    onNavigateToBacterialScan: () -> Unit = {},
    onNavigateToBacterialHistory: () -> Unit = {},
    viewModel: DashboardViewModel = viewModel()
) {
    // ViewModel state'lerini collect et
    val pendingUploads by viewModel.pendingUploads.collectAsState()
    val selectedTimeFrame by viewModel.selectedTimeFrame.collectAsState()
    val analysisSummary by viewModel.analysisSummary.collectAsState()
    val errorMessage by viewModel.errorMessage.collectAsState()
    val showError by viewModel.showError.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()

    var showDeveloperOptions by remember { mutableStateOf(false) }

    // SwiftUI referansına göre stat cards oluştur
    val statCards by remember(analysisSummary, pendingUploads) {
        derivedStateOf {
            val summary = analysisSummary
            listOf(
                StatCardData(
                    title = "Total Analyses",
                    value = summary?.totalCount?.toString() ?: "0",
                    icon = Icons.Default.Assessment,
                    color = Color(0xFF007AFF)
                ),
                StatCardData(
                    title = "Pending Uploads",
                    value = pendingUploads.toString(),
                    icon = Icons.AutoMirrored.Filled.Send,
                    color = if (pendingUploads > 0) Color(0xFFFF9500) else Color(0xFF34C759)
                ),
                StatCardData(
                    title = "Most Common",
                    value = summary?.parasiteCounts?.maxByOrNull { it.value }?.key?.displayName ?: "None",
                    icon = Icons.AutoMirrored.Filled.TrendingUp,
                    color = Color(0xFF5856D6)
                ),
                StatCardData(
                    title = "Infection Rate",
                    value = "${((summary?.parasiteCounts?.values?.sum() ?: 0) * 100 / maxOf(summary?.totalCount ?: 1, 1))}%",
                    icon = Icons.Default.Analytics,
                    color = Color(0xFFAF52DE)
                )
            )
        }
    }

    // Upload button animasyonu
    val uploadButtonScale by animateFloatAsState(
        targetValue = if (pendingUploads > 0) 1.1f else 1f,
        animationSpec = tween(durationMillis = 300),
        label = ""
    )

    // Error handling
    if (showError && errorMessage != null) {
        val currentErrorMessage = errorMessage
        AlertDialog(
            onDismissRequest = { viewModel.dismissError() },
            title = { Text("Error") },
            text = { Text(currentErrorMessage ?: "") },
            confirmButton = {
                TextButton(onClick = { viewModel.dismissError() }) {
                    Text("OK")
                }
            }
        )
    }

    Scaffold(
        topBar = {
            DashboardTopBar(
                pendingUploads = pendingUploads,
                showDeveloperOptions = showDeveloperOptions,
                onDeveloperOptionsToggle = { showDeveloperOptions = !showDeveloperOptions },
                uploadButtonScale = uploadButtonScale,
                onUploadClick = { viewModel.uploadPendingAnalyses() },
                isLoading = isLoading
            )
        }
    ) { paddingValues ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues),
            contentPadding = PaddingValues(20.dp),
            verticalArrangement = Arrangement.spacedBy(20.dp)
        ) {
            // ML Test Buttons
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Button(
                            onClick = onNavigateToMnist,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("MNIST Rakam Tanıma Testi")
                        }
                        Button(
                            onClick = onNavigateToBacterialTest,
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.tertiary
                            )
                        ) {
                            Icon(Icons.Default.BugReport, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text("Bacterial Test (Debug)")
                        }
                        Button(
                            onClick = onNavigateToBacterialScan,
                            modifier = Modifier.fillMaxWidth(),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.secondary
                            )
                        ) {
                            Icon(Icons.Default.Science, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text("Bacterial Colony Scan")
                        }
                        OutlinedButton(
                            onClick = onNavigateToBacterialHistory,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(Icons.Default.History, contentDescription = null)
                            Spacer(Modifier.width(8.dp))
                            Text("View Bacterial History")
                        }
                    }
                }
            }
            
            // Stats section - SwiftUI'deki statsView gibi
            item {
                StatsSection(
                    statCards = statCards,
                    selectedTimeFrame = selectedTimeFrame,
                    onTimeFrameChanged = { viewModel.updateTimeFrame(it) }
                )
            }

            // Parasite distribution section - SwiftUI'deki parasiteDistributionView gibi
            if (analysisSummary != null) {
                item {
                    ParasiteDistributionSection(analysisSummary = analysisSummary!!)
                }
            }

            // Pending uploads section - SwiftUI'deki uploadSection gibi
            if (pendingUploads > 0) {
                item {
                    PendingUploadsSection(
                        pendingCount = pendingUploads,
                        onUploadClick = { viewModel.uploadPendingAnalyses() },
                        isLoading = isLoading
                    )
                }
            }

            // Developer Options - SwiftUI'deki developerOptionsView gibi
            if (showDeveloperOptions) {
                item {
                    DeveloperOptionsSection(
                        onNavigateToCamera = onNavigateToCamera,
                        onNavigateToHistory = onNavigateToHistory,
                        onNavigateToProfile = onNavigateToProfile
                    )
                }
            }

            // Recent analyses section - SwiftUI'deki recentAnalysesView gibi
            item {
                RecentAnalysesSection(
                    onAnalysisClick = onAnalysisClick,
                    recentAnalyses = emptyList() // TODO: ViewModel'den gerçek data
                )
            }
        }
    }

    // Error handling
    if (showError && errorMessage != null) {
        val currentErrorMessage = errorMessage
        AlertDialog(
            onDismissRequest = { viewModel.dismissError() },
            title = { Text("Error") },
            text = { Text(currentErrorMessage ?: "") },
            confirmButton = {
                TextButton(onClick = { viewModel.dismissError() }) {
                    Text("OK")
                }
            }
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun DashboardTopBar(
    pendingUploads: Int,
    showDeveloperOptions: Boolean,
    onDeveloperOptionsToggle: () -> Unit,
    uploadButtonScale: Float,
    onUploadClick: () -> Unit,
    isLoading: Boolean
) {
    TopAppBar(
        title = {
            Text(
                "Dashboard",
                fontWeight = FontWeight.SemiBold
            )
        },
        actions = {
            if (pendingUploads > 0) {
                IconButton(
                    onClick = onUploadClick,
                    modifier = Modifier.scale(uploadButtonScale),
                    enabled = !isLoading
                ) {
                    if (isLoading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(24.dp),
                            strokeWidth = 2.dp
                        )
                    } else {
                        BadgedBox(
                            badge = {
                                Badge {
                                    Text(pendingUploads.toString())
                                }
                            }
                        ) {
                            Icon(
                                Icons.AutoMirrored.Filled.Send,
                                contentDescription = "Upload Pending Analyses"
                            )
                        }
                    }
                }
            }
        },
        navigationIcon = {
            IconButton(onClick = onDeveloperOptionsToggle) {
                Icon(
                    Icons.Default.Build,
                    contentDescription = "Developer Options",
                    tint = if (showDeveloperOptions)
                        MaterialTheme.colorScheme.primary
                    else
                        MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    )
}

@Composable
private fun StatsSection(
    statCards: List<StatCardData>,
    selectedTimeFrame: DashboardViewModel.TimeFrame,
    onTimeFrameChanged: (DashboardViewModel.TimeFrame) -> Unit
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header with time frame picker - SwiftUI'deki HStack gibi
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "Statistics",
                fontSize = 20.sp,
                fontWeight = FontWeight.SemiBold
            )

            // Time frame picker - SwiftUI'deki Picker gibi
            var expanded by remember { mutableStateOf(false) }

            Box {
                OutlinedButton(
                    onClick = { expanded = true },
                    shape = RoundedCornerShape(20.dp)
                ) {
                    Text(
                        selectedTimeFrame.displayName,
                        fontSize = 14.sp
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Icon(
                        Icons.Default.ArrowDropDown,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp)
                    )
                }

                DropdownMenu(
                    expanded = expanded,
                    onDismissRequest = { expanded = false }
                ) {
                    DashboardViewModel.TimeFrame.entries.forEach { timeFrame ->
                        DropdownMenuItem(
                            text = { Text(timeFrame.displayName) },
                            onClick = {
                                onTimeFrameChanged(timeFrame)
                                expanded = false
                            }
                        )
                    }
                }
            }
        }

        // Stats grid - SwiftUI'deki LazyVGrid gibi
        LazyVerticalGrid(
            columns = GridCells.Fixed(2),
            modifier = Modifier.height(200.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            items(statCards) { statCard ->
                StatCard(statCard = statCard)
            }
        }
    }
}

@Composable
private fun StatCard(statCard: StatCardData) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(90.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top
            ) {
                Icon(
                    imageVector = statCard.icon,
                    contentDescription = null,
                    tint = statCard.color,
                    modifier = Modifier.size(24.dp)
                )
            }

            Column {
                Text(
                    text = statCard.value,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface
                )
                Text(
                    text = statCard.title,
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

// SwiftUI'deki parasiteDistributionView benzeri
@Composable
private fun ParasiteDistributionSection(analysisSummary: DashboardViewModel.AnalysisSummary) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = "Parasite Distribution",
                fontSize = 18.sp,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(bottom = 16.dp)
            )

            // Parasite counts list
            analysisSummary.parasiteCounts.entries.forEach { (parasiteType, count) ->
                if (count > 0) {
                    ParasiteCountRow(
                        parasiteType = parasiteType,
                        count = count,
                        total = analysisSummary.totalCount
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
    }
}

@Composable
private fun ParasiteCountRow(
    parasiteType: ParasiteType,
    count: Int,
    total: Int
) {
    val percentage = if (total > 0) (count * 100f / total) else 0f

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = parasiteType.displayName,
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium
            )
            LinearProgressIndicator(
                progress = { percentage / 100f },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp),
                color = when (parasiteType) {
                    ParasiteType.ASCARIS -> Color(0xFF007AFF)
                    ParasiteType.HOOKWORM -> Color(0xFF34C759)
                    ParasiteType.TRICHURIS -> Color(0xFFFF9500)
                    ParasiteType.HYMENOLEPIS -> Color(0xFFAF52DE)
                },
                trackColor = MaterialTheme.colorScheme.surfaceVariant,
            )
        }

        Spacer(modifier = Modifier.width(16.dp))

        Text(
            text = "$count (${percentage.toInt()}%)",
            fontSize = 12.sp,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

// SwiftUI'deki uploadSection benzeri
@Composable
private fun PendingUploadsSection(
    pendingCount: Int,
    onUploadClick: () -> Unit,
    isLoading: Boolean
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFFFF9500).copy(alpha = 0.1f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "Pending Uploads",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "$pendingCount analyses waiting to upload",
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                Button(
                    onClick = onUploadClick,
                    enabled = !isLoading,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFFF9500)
                    )
                ) {
                    if (isLoading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            strokeWidth = 2.dp,
                            color = Color.White
                        )
                    } else {
                        Icon(
                            Icons.AutoMirrored.Filled.Send,
                            contentDescription = null,
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Upload")
                    }
                }
            }
        }
    }
}

// SwiftUI'deki VisualizationSection benzeri
@Composable
private fun VisualizationSection() {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
        ) {
            Text(
                text = "Analysis Trends",
                fontSize = 18.sp,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(bottom = 16.dp)
            )

            // Placeholder for chart - gerçek chart kütüphanesi sonra eklenecek
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f),
                        RoundedCornerShape(12.dp)
                    ),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Icon(
                        Icons.AutoMirrored.Filled.ShowChart,
                        contentDescription = null,
                        modifier = Modifier.size(48.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Chart Visualization",
                        fontSize = 14.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                    )
                    Text(
                        text = "(Coming Soon)",
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f)
                    )
                }
            }
        }
    }
}

@Composable
private fun DeveloperOptionsSection(
    onNavigateToCamera: () -> Unit,
    onNavigateToHistory: () -> Unit,
    onNavigateToProfile: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.1f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    Icons.Default.Build,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.error,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Developer Options",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.error
                )
            }

            Text(
                text = "Quick access to app features for testing and development",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            // Navigation buttons grid
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedButton(
                    onClick = onNavigateToCamera,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = MaterialTheme.colorScheme.error
                    )
                ) {
                    Icon(
                        Icons.Default.CameraAlt,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Camera", fontSize = 12.sp)
                }
                OutlinedButton(
                    onClick = onNavigateToHistory,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = MaterialTheme.colorScheme.error
                    )
                ) {
                    Icon(
                        Icons.Default.History,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("History", fontSize = 12.sp)
                }
                OutlinedButton(
                    onClick = onNavigateToProfile,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.outlinedButtonColors(
                        contentColor = MaterialTheme.colorScheme.error
                    )
                ) {
                    Icon(
                        Icons.Default.Person,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Profile", fontSize = 12.sp)
                }
            }
        }
    }
}

@Composable
private fun RecentAnalysesSection(
    onAnalysisClick: (String) -> Unit,
    recentAnalyses: List<Analysis>
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        shape = RoundedCornerShape(16.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Recent Analyses",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold
                )

                TextButton(
                    onClick = { /* Navigate to full history */ }
                ) {
                    Text("View All")
                    Icon(
                        Icons.AutoMirrored.Filled.ArrowForward,
                        contentDescription = null,
                        modifier = Modifier.size(16.dp)
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            if (recentAnalyses.isEmpty()) {
                // Empty state - SwiftUI'deki empty state benzeri
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(120.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        Icon(
                            Icons.Default.Analytics,
                            contentDescription = null,
                            modifier = Modifier.size(48.dp),
                            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "No Recent Analyses",
                            fontSize = 16.sp,
                            fontWeight = FontWeight.Medium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                        Text(
                            text = "Start by taking your first scan",
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f)
                        )
                    }
                }
            } else {
                // Recent analyses list
                LazyColumn(
                    modifier = Modifier.height(200.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(recentAnalyses) { analysis ->
                        RecentAnalysisItem(
                            analysis = analysis,
                            onClick = { onAnalysisClick(analysis.id) }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun RecentAnalysisItem(
    analysis: Analysis,
    onClick: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick() },
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
        shape = RoundedCornerShape(12.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(
                modifier = Modifier.weight(1f),
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Analysis type icon
                Icon(
                    imageVector = when (analysis.analysisType) {
                        AnalysisType.PARASITE -> Icons.Default.BugReport
                        AnalysisType.BLOOD -> Icons.Default.LocalHospital
                        AnalysisType.URINE -> Icons.Default.Science
                        else -> Icons.Default.Science // else branch eklendi
                    },
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(20.dp)
                )

                Spacer(modifier = Modifier.width(12.dp))

                Column {
                    Text(
                        text = analysis.dominantParasite?.displayName ?: "Unknown",
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Medium
                    )
                    Text(
                        text = SimpleDateFormat("MMM dd, HH:mm", Locale.getDefault()).format(analysis.timestamp),
                        fontSize = 12.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            // Upload status
            if (!analysis.isUploaded) {
                Icon(
                    Icons.Default.CloudUpload,
                    contentDescription = "Pending upload",
                    tint = Color(0xFFFF9500),
                    modifier = Modifier.size(16.dp)
                )
            } else {
                Icon(
                    Icons.Default.CloudDone,
                    contentDescription = "Uploaded",
                    tint = Color(0xFF34C759),
                    modifier = Modifier.size(16.dp)
                )
            }
        }
    }
}

// MARK: - Preview Functions (Swift benzeri)
@Preview(showBackground = true)
@Composable
fun DashboardScreenPreview() {
    VisionVetAITheme {
        DashboardScreen(
            onNavigateToCamera = {},
            onNavigateToHistory = {},
            onNavigateToProfile = {},
            onAnalysisClick = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun DashboardScreenDarkPreview() {
    VisionVetAITheme(darkTheme = true) {
        DashboardScreen(
            onNavigateToCamera = {},
            onNavigateToHistory = {},
            onNavigateToProfile = {},
            onAnalysisClick = {}
        )
    }
}

@Preview(showBackground = true, widthDp = 320, heightDp = 568)
@Composable
fun DashboardScreenSmallDevicePreview() {
    VisionVetAITheme {
        DashboardScreen(
            onNavigateToCamera = {},
            onNavigateToHistory = {},
            onNavigateToProfile = {},
            onAnalysisClick = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun StatCardPreview() {
    VisionVetAITheme {
        StatCard(
            statCard = StatCardData(
                title = "Total Scans",
                value = "47",
                icon = Icons.Default.Info,
                color = Color(0xFF007AFF)
            )
        )
    }
}

@Preview(showBackground = true)
@Composable
fun StatsSectionPreview() {
    VisionVetAITheme {
        StatsSection(
            statCards = listOf(
                StatCardData(
                    title = "Total Scans",
                    value = "47",
                    icon = Icons.Default.Info,
                    color = Color(0xFF007AFF)
                ),
                StatCardData(
                    title = "Pending Uploads",
                    value = "3",
                    icon = Icons.AutoMirrored.Filled.Send,
                    color = Color(0xFFFF9500)
                )
            ),
            selectedTimeFrame = DashboardViewModel.TimeFrame.WEEK,
            onTimeFrameChanged = {}
        )
    }
}

@Preview(showBackground = true)
@Composable
fun VisualizationSectionPreview() {
    VisionVetAITheme {
        VisualizationSection()
    }
}

@Preview(showBackground = true)
@Composable
fun DeveloperOptionsSectionPreview() {
    VisionVetAITheme {
        DeveloperOptionsSection(
            onNavigateToCamera = {},
            onNavigateToHistory = {},
            onNavigateToProfile = {}
        )
    }
}

// MARK: - Simplified Preview Functions
@Preview(showBackground = true, showSystemUi = true)
@Composable
fun DashboardScreenSimplePreview() {
    VisionVetAITheme {
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Header
                Text(
                    text = "Dashboard",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold
                )

                // Stats Cards Preview
                LazyVerticalGrid(
                    columns = GridCells.Fixed(2),
                    modifier = Modifier.height(200.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(4) { index ->
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(90.dp),
                            colors = CardDefaults.cardColors(
                                containerColor = MaterialTheme.colorScheme.surface
                            ),
                            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
                            shape = RoundedCornerShape(16.dp)
                        ) {
                            Column(
                                modifier = Modifier
                                    .fillMaxSize()
                                    .padding(16.dp),
                                verticalArrangement = Arrangement.SpaceBetween
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Assessment,
                                    contentDescription = null,
                                    tint = Color(0xFF007AFF),
                                    modifier = Modifier.size(24.dp)
                                )

                                Column {
                                    Text(
                                        text = when(index) {
                                            0 -> "11"
                                            1 -> "2"
                                            2 -> "Ascaris"
                                            else -> "45%"
                                        },
                                        fontSize = 18.sp,
                                        fontWeight = FontWeight.Bold
                                    )
                                    Text(
                                        text = when(index) {
                                            0 -> "Total Analyses"
                                            1 -> "Pending Uploads"
                                            2 -> "Most Common"
                                            else -> "Infection Rate"
                                        },
                                        fontSize = 12.sp,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                        }
                    }
                }

                // Sample Card
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surface
                    ),
                    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            text = "Parasite Distribution",
                            fontSize = 18.sp,
                            fontWeight = FontWeight.SemiBold,
                            modifier = Modifier.padding(bottom = 16.dp)
                        )

                        // Sample progress bars
                        repeat(3) { index ->
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 4.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = when(index) {
                                            0 -> "Ascaris"
                                            1 -> "Hookworm"
                                            else -> "Trichuris"
                                        },
                                        fontSize = 14.sp,
                                        fontWeight = FontWeight.Medium
                                    )
                                    LinearProgressIndicator(
                                        progress = { when(index) { 0 -> 0.7f; 1 -> 0.4f; else -> 0.2f } },
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .height(6.dp),
                                        color = when(index) {
                                            0 -> Color(0xFF007AFF)
                                            1 -> Color(0xFF34C759)
                                            else -> Color(0xFFFF9500)
                                        },
                                        trackColor = MaterialTheme.colorScheme.surfaceVariant,
                                    )
                                }

                                Text(
                                    text = when(index) { 0 -> "7 (70%)"; 1 -> "4 (40%)"; else -> "2 (20%)" },
                                    fontSize = 12.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    modifier = Modifier.padding(start = 16.dp)
                                )
                            }

                            if (index < 2) {
                                Spacer(modifier = Modifier.height(8.dp))
                            }
                        }
                    }
                }
            }
        }
    }
}
